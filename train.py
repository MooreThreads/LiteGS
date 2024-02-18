import sys
from argparse import ArgumentParser, Namespace
from training.arguments import ModelParams,OptimizationParams,PipelineParams
from loader import TrainingDataLoader
import typing
from loader.InfoLoader import CameraInfo,ImageInfo
from gaussian_splatting.gaussian_util import GaussianScene
from gaussian_splatting.model import GaussianSplattingModel
from training.training import GaussianTrain

if __name__ == "__main__":
    # Init Training Arg
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[4_000,7_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[4_000,7_000, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    lp:ModelParams=lp.extract(args)
    op:OptimizationParams=op.extract(args)
    pp:PipelineParams=pp.extract(args)

    #load training data
    cameras_info:typing.Dict[int,CameraInfo]=None
    images_info:typing.List[ImageInfo]=None
    scene:GaussianScene=None
    cameras_info,images_info,scene,_,NerfNormRadius=TrainingDataLoader.load(lp.source_path,lp.images,lp.sh_degree)

    #params & optimizer
    gaussian_model=GaussianSplattingModel(None,scene,NerfNormRadius)
    training=GaussianTrain(gaussian_model,op,NerfNormRadius,images_info,cameras_info)

    #start
    training.start(op.iterations)
