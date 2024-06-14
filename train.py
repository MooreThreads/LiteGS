import sys
from argparse import ArgumentParser, Namespace
from training.arguments import ModelParams,OptimizationParams,PipelineParams
from loader import TrainingDataLoader
import typing
from loader.InfoLoader import CameraInfo,ImageInfo
from gaussian_splatting.scene import GaussianScene
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
    parser.add_argument("--test_epochs", nargs="+", type=int, default=[17,31])
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[50,100,186])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[50,100,186])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_epochs.append(args.epoch)
    lp:ModelParams=lp.extract(args)
    op:OptimizationParams=op.extract(args)
    pp:PipelineParams=pp.extract(args)

    #load training data
    cameras_info:typing.Dict[int,CameraInfo]=None
    images_info:typing.List[ImageInfo]=None
    scene:GaussianScene=None
    cameras_info,images_info,scene,_,NerfNormRadius=TrainingDataLoader.load(lp.source_path,lp.images,lp.sh_degree,lp.resolution)

    #params & optimizer
    gaussian_model=GaussianSplattingModel(scene,NerfNormRadius)
    training=GaussianTrain(gaussian_model,lp,op,NerfNormRadius,images_info,cameras_info)

    #start
    training.start(op.epoch,args.start_checkpoint,args.checkpoint_epochs,args.save_epochs,args.test_epochs)
    #training.torch_profiler(1)
