from argparse import ArgumentParser, Namespace
import torch
import sys

import litegs
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = litegs.arguments.ModelParams(parser)
    op = litegs.arguments.OptimizationParams(parser)
    pp = litegs.arguments.PipelineParams(parser)
    
    parser.add_argument("--test_epochs", nargs="+", type=int, default=[139,199])
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    lp:litegs.arguments.ModelParams=lp.extract(args)
    op:litegs.arguments.OptimizationParams=op.extract(args)
    pp:litegs.arguments.PipelineParams=pp.extract(args)

    litegs.training.start(lp,op,pp,args.start_checkpoint)