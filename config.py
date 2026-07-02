"""
Unified Configuration for InvertedManifoldEngine + HashGAN + CIFAR Pipeline
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    G_ARCHITECTURE: str = "NORM"
    D_ARCHITECTURE: str = "NORM"
    DIM_G: int = 128
    DIM_D: int = 128
    DIM: int = 64
    HASH_DIM: int = 64
    G_PRETRAINED_MODEL_PATH: str = ""
    D_PRETRAINED_MODEL_PATH: str = ""
    ALEXNET_PRETRAINED_MODEL_PATH: str = "./pretrained_models/reference_pretrain.npy"


@dataclass
class DataConfig:
    """Dataset and data loading configuration."""
    USE_DATASET: str = "cifar10"
    LABEL_DIM: int = 10
    DB_SIZE: int = 54000
    TEST_SIZE: int = 1000
    WIDTH_HEIGHT: int = 32
    OUTPUT_DIM: int = 3072
    MAP_R: int = 54000
    LIST_ROOT: str = "./data_list/cifar10"
    DATA_ROOT: str = "./data/cifar10"
    OUTPUT_DIR: str = "./output/cifar10_step_1"
    IMAGE_DIR: str = ""
    MODEL_DIR: str = ""
    LOG_DIR: str = ""

    def __post_init__(self):
        self.OUTPUT_DIM = 3 * (self.WIDTH_HEIGHT ** 2)
        if not self.IMAGE_DIR:
            self.IMAGE_DIR = os.path.join(self.OUTPUT_DIR, "images")
        if not self.MODEL_DIR:
            self.MODEL_DIR = os.path.join(self.OUTPUT_DIR, "models")
        if not self.LOG_DIR:
            self.LOG_DIR = os.path.join(self.OUTPUT_DIR, "logs")


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    EVALUATE_MODE: bool = False
    BATCH_SIZE: int = 64
    ITERS: int = 100000
    CROSS_ENTROPY_ALPHA: float = 5.0
    LR: float = 1e-4
    G_LR: float = 1e-4
    DECAY: bool = True
    N_CRITIC: int = 5
    EVAL_FREQUENCY: int = 20000
    CHECKPOINT_FREQUENCY: int = 2000
    SAMPLE_FREQUENCY: int = 1000
    ACGAN_SCALE: float = 1.0
    ACGAN_SCALE_FAKE: float = 1.0
    WGAN_SCALE: float = 1.0
    WGAN_SCALE_GP: float = 10.0
    ACGAN_SCALE_G: float = 0.1
    WGAN_SCALE_G: float = 1.0
    NORMED_CROSS_ENTROPY: bool = True
    FAKE_RATIO: float = 1.0


@dataclass
class EngineConfig:
    """InvertedManifoldEngine hyperparameters."""
    ENABLED: bool = True
    TAU: float = 0.01
    LAM: float = 1.0
    ALPHA_PARAM: float = 5.0
    ENTROPY_THRESHOLD: float = 1.45
    BETA: float = 0.1


@dataclass
class Config:
    """Top-level configuration container."""
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    DATA: DataConfig = field(default_factory=DataConfig)
    TRAIN: TrainConfig = field(default_factory=TrainConfig)
    ENGINE: EngineConfig = field(default_factory=EngineConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        model_cfg = ModelConfig(**d.get("MODEL", {}))
        data_cfg = DataConfig(**d.get("DATA", {}))
        train_cfg = TrainConfig(**d.get("TRAIN", {}))
        engine_cfg = EngineConfig(**d.get("ENGINE", {}))
        return cls(MODEL=model_cfg, DATA=data_cfg,
                    TRAIN=train_cfg, ENGINE=engine_cfg)

    def ensure_dirs(self):
        for d in [self.DATA.IMAGE_DIR, self.DATA.MODEL_DIR, self.DATA.LOG_DIR]:
            os.makedirs(d, exist_ok=True)

    def to_dict(self) -> dict:
        result = {}
        for section in ["MODEL", "DATA", "TRAIN", "ENGINE"]:
            section_obj = getattr(self, section)
            result[section] = {
                k: v for k, v in section_obj.__dict__.items()
                if not k.startswith('_')
            }
        return result


def cifar_step1_config():
    cfg = Config()
    cfg.DATA.OUTPUT_DIR = "./output/cifar10_step_1"
    cfg.TRAIN.ITERS = 22000
    cfg.TRAIN.G_LR = 1e-4
    cfg.TRAIN.ACGAN_SCALE_G = 1.0
    cfg.TRAIN.WGAN_SCALE_G = 1.0
    cfg.TRAIN.ACGAN_SCALE_FAKE = 0.0
    cfg.DATA.__post_init__()
    return cfg


def cifar_step2_config():
    cfg = Config()
    cfg.DATA.OUTPUT_DIR = "./output/cifar10_step_2"
    cfg.TRAIN.ITERS = 50000
    cfg.TRAIN.G_LR = 1e-4
    cfg.TRAIN.N_CRITIC = 1
    cfg.TRAIN.ACGAN_SCALE_FAKE = 1.0
    cfg.MODEL.G_PRETRAINED_MODEL_PATH = "./output/cifar10_step_1/models/G_21999.ckpt"
    cfg.DATA.__post_init__()
    return cfg


def cifar_eval_config():
    cfg = Config()
    cfg.DATA.OUTPUT_DIR = "./output/cifar10_evaluation"
    cfg.TRAIN.EVALUATE_MODE = True
    cfg.TRAIN.BATCH_SIZE = 128
    cfg.TRAIN.ITERS = 10000
    cfg.TRAIN.G_LR = 0.0
    cfg.TRAIN.ACGAN_SCALE = 1.0
    cfg.TRAIN.ACGAN_SCALE_FAKE = 0.0
    cfg.TRAIN.WGAN_SCALE = 0.0
    cfg.TRAIN.ACGAN_SCALE_G = 0.1
    cfg.TRAIN.WGAN_SCALE_G = 1.0
    cfg.MODEL.G_PRETRAINED_MODEL_PATH = "./output/cifar10_step_1/models/G_21999.ckpt"
    cfg.MODEL.D_PRETRAINED_MODEL_PATH = "./output/cifar10_finetune/models/D_9999.ckpt"
    cfg.DATA.__post_init__()
    return cfg
