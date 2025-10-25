import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    T2IAdapter,
    UNet2DConditionModel,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import ImageMaskDataset


# ========== モデルのセットアップ ==========
def create_models(
    config: Config,
) -> tuple[T2IAdapter, UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler]:
    adapter = T2IAdapter(**config.model.model_dump())
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    # Cross-Attention Block無効化
    for name, module in unet.named_modules():
        if hasattr(module, "attn2"):  # SD2.1 uses attn2 for encoder attention
            module.attn2 = None

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    scheduler = EulerDiscreteScheduler.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler"
    )
    return adapter, unet, vae, scheduler


# ========== トレーニングループ ==========
def main():
    assert torch.cuda.is_available()
    config = Config.from_json("./configs/defaults.json")
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    adapter, unet, vae, scheduler = create_models(config)
    adapter.to(device)
    unet.to(device)
    vae.to(device)
    unet.eval()
    vae.eval()

    dataset = ImageMaskDataset(**config.dataset.model_dump())
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-4)
    adapter, optimizer, dataloader = accelerator.prepare(adapter, optimizer, dataloader)

    unet_dtype = next(unet.parameters()).dtype

    for epoch in range(5):
        pbar = tqdm(dataloader)
        for images, masks in pbar:
            images = images.to(device).half()
            masks = masks.to(device).half()
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = (latents * 0.18215).to(unet.dtype)

            adapter_features = adapter(masks)  # (B, feats, H, W)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (latents.size(0),),
                device=latents.device,
            )
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # UNet: prompt無 → text_embeds = None
            with torch.amp.autocast(str(device), dtype=unet_dtype):
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=None,
                    down_intrablock_additional_residuals=adapter_features,
                ).sample

            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description_str(f"Loss: {loss.item():.4f}")

    accelerator.wait_for_everyone()
    accelerator.print("Training finished.")
    accelerator.save_state("t2i_adapter_checkpoint")


if __name__ == "__main__":
    main()
