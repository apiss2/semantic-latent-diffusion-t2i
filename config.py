from pydantic import BaseModel, model_validator
from pathlib import Path
import json


class ModelConfig(BaseModel):
    # T2IAdaptor
    in_channels: int = 3
    channels: list[int] = [320, 640, 1280, 1280]
    num_res_blocks: int = 2
    downscale_factor: int = 8
    adapter_type: str = "full_adapter"


class DatasetConfig(BaseModel):
    image_dir: str = "path_to_imagedir"
    mask_dir: str = "path_to_maskdir"
    size: int = 512
    image_suffix: str = ".jpg"
    mask_suffix: str = ".png"


class Config(BaseModel):
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = DatasetConfig()

    # ====================== Validation Rules (Was ValueError Part) ======================
    @model_validator(mode="after")
    def check_dataset_or_train_dir(cls, values):
        # if values.dataset_name is None and values.train_data_dir is None:
        #     raise ValueError("Specify either `dataset_name` or `train_data_dir`")
        return values

    # ====================== JSON Utility ======================
    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=4))
