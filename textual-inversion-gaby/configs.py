from pydantic import BaseModel
import yaml


class TrainingConfig(BaseModel):
    lr: float
    train_steps: int
    bs: int
    log_each: int
    repeats: int


class TextualInversionConfig(BaseModel):
    name: str
    data_dir: str
    sd_checkpoint_name: str
    placeholder_token: str
    initializer_token: str
    evaluation_prompt: str
    train: TrainingConfig


def load(file_path: str) -> TextualInversionConfig:
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return TextualInversionConfig(**config)


if __name__ == "__main__":
    config = load("./embeddings/baseline.yaml")
    print(config)
