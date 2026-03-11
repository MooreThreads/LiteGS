import tyro
from mashumaro.mixins.json import DataClassJSONMixin
from dataclasses import dataclass, field
# 【核心修复】：把 Callable, Union, Type, Any 全部加上，喂给底层的解析器
from typing import Tuple, List, Optional, Annotated
import argparse

# ==========================================
# 1. Dataset & System Environment
# ==========================================
@dataclass
class DatasetParams:
    """Dataset and System Environment Configuration"""
    
    # Path to the input source data
    source_path: Annotated[str, tyro.conf.arg(aliases=["-s"])] = ""
    # Path to save the trained model checkpoints
    model_path: Annotated[str, tyro.conf.arg(aliases=["-m"])] = ""
    # Name of the images folder within the source path
    images: Annotated[str, tyro.conf.arg(aliases=["-i"])] = "images"
    # Resolution scale for loading images (-1 for original resolution)
    resolution: Annotated[int, tyro.conf.arg(aliases=["-r"])] = -1
    # PyTorch device for data
    data_device: str = "cuda"
    # Whether to split the dataset for train/test evaluation
    eval: bool = False
    # Whether to preload all data directly into GPU memory
    device_preload: bool = True

# ==========================================
# 2. Rendering Pipeline Configuration
# ==========================================
@dataclass
class PipelineParams:
    """Rendering Pipeline Configuration"""

    # Use a white background instead of black
    white_background: Annotated[bool, tyro.conf.arg(aliases=["-w"])] = False
    # Enable rendering of the transmittance map
    enable_transmitance: bool = False
    # Enable rendering of the depth map
    enable_depth: bool = False
    # Tile size for rasterization (height, width)
    tile_size: Tuple[int, int] = (8, 16)
    # Whether to optimize camera poses (view and projection matrices)
    learnable_viewproj: bool = False

# ==========================================
# 3. 3D Asset (Gaussian Model) Configuration
# ==========================================
@dataclass
class ModelParams:
    """3D Asset (Gaussian Model) Configuration"""

    # Maximum degree of Spherical Harmonics (SH)
    sh_degree: int = 3
    # Size of the chunks/clusters. Set to 0 to disable clustering (Vanilla GS)
    cluster_size: int = 128
    # Representation of the input color ('rgb' or 'sh')
    input_color_type: str = 'sh'

# ==========================================
# 4. Optimization & Densification
# ==========================================
@dataclass
class OptimizationParams:
    """Optimization Parameters"""

    # Total number of training iterations
    iterations: int = 30000
    # Initial learning rate for positions (xyz)
    position_lr_init: float = 0.00016
    # Final learning rate for positions (xyz)
    position_lr_final: float = 0.0000016
    # Maximum steps for position learning rate scheduling
    position_lr_max_steps: int = 30000
    # Learning rate for Spherical Harmonics features
    feature_lr: float = 0.0025
    # Learning rate for opacity
    opacity_lr: float = 0.025
    # Learning rate for scaling factors
    scaling_lr: float = 0.005
    # Learning rate for rotation quaternions
    rotation_lr: float = 0.001
    # Weight for the DSSIM loss term
    lambda_dssim: float = 0.2
    # Whether to use sparse gradients for position and features
    sparse_grad: bool = True

@dataclass
class DensifyParams:
    """Densification and Pruning Parameters"""

    # Interval (in epochs) between densification steps
    interval: int = 5
    # Epoch to start densification
    start: int = 3
    # Epoch to stop densification (-1 for auto-calc)
    end: int = -1
    # Interval (in epochs) for resetting/decaying opacity
    opacity_reset_interval: int = 10
    # Mode for opacity reset ('decay' or 'reset')
    opacity_reset_mode: str = 'decay'
    # Criteria for pruning ('weight' or 'threshold')
    prune_mode: str = 'weight'
    # Target maximum number of Gaussian primitives
    target_primitives: int = 1000000
    
    # 【修复 2】必须加上类型注解，否则 Dataclass 无法识别它们作为字段！
    # [discard] params for official densification(Adaptive Density Control)
    densify_grad_threshold: float = 0.00015
    # [discard] params for official densification(Adaptive Density Control)
    opacity_threshold: float = 0.005
    # [discard] params for official densification(Adaptive Density Control)
    screen_size_threshold: int = 128
    # [discard] params for official densification(Adaptive Density Control)
    percent_dense: float = 0.01

# ==========================================
# 5. Global Config Entry
# ==========================================
@dataclass
class TrainConfig(DataClassJSONMixin):
    """Global Training Configuration for 3D Gaussian Splatting"""

    dataset: DatasetParams = field(default_factory=DatasetParams)
    pipeline: PipelineParams = field(default_factory=PipelineParams)
    model: ModelParams = field(default_factory=ModelParams)
    opt: OptimizationParams = field(default_factory=OptimizationParams)
    densify: DensifyParams = field(default_factory=DensifyParams)

    # Path of config file
    config_path: Optional[str] = None
    # List of epochs to evaluate on the test set
    test_epochs: List[int] = field(default_factory=list)
    # List of epochs to save the point cloud (.ply) models
    save_epochs: List[int] = field(default_factory=list)
    # List of epochs to save full training checkpoints
    cp_epochs: List[int] = field(default_factory=list)
    # Path to a checkpoint (.pth) to resume training from
    resume: Optional[str] = None

@dataclass
class EvalConfig(DataClassJSONMixin):
    """Global Evaluation Configuration for 3D Gaussian Splatting"""
    
    dataset: DatasetParams = field(default_factory=DatasetParams)
    pipeline: PipelineParams = field(default_factory=PipelineParams)
    model: ModelParams = field(default_factory=ModelParams)
    
    # Whether to save rendered images to disk during evaluation
    save_image: bool = False
    # Path of config file
    config_path: Optional[str] = None


def get_config(cls:DataClassJSONMixin):
    
    # Get config_path
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-path", type=str, default=None)
    args, remaining_args = pre_parser.parse_known_args()
    
    # Deserialization
    default_config=None
    if args.config_path is not None:
        with open(args.config_path, "r") as f:
            default_config = cls.from_json(f.read())
            print(f"[Config] Successfully loaded configuration from: {args.config_path}")

    
    cfg = tyro.cli(cls, default=default_config, args=remaining_args)
    return cfg