from argparse import ArgumentParser, Namespace
import torch
import sys

import litegs
if __name__ == "__main__":
    # test_param=torch.nn.Parameter(torch.ones((1120,128)))
    # opt=torch.optim.Adam([test_param,])
    # for i in range(100):
    #     test_param.sum().backward()
    #     opt.step()
    #     opt.zero_grad(set_to_none=True)
    # test_param.data=test_param[:1000]
    # test_param.sum().backward()



    parser = ArgumentParser(description="Training script parameters")
    lp = litegs.arguments.ModelParams(parser)
    op = litegs.arguments.OptimizationParams(parser)
    pp = litegs.arguments.PipelineParams(parser)
    dp = litegs.arguments.DensifyParams(parser)
    
    parser.add_argument("--test_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    lp:litegs.arguments.ModelParams=lp.extract(args)
    op:litegs.arguments.OptimizationParams=op.extract(args)
    pp:litegs.arguments.PipelineParams=pp.extract(args)
    dp:litegs.arguments.DensifyParams=dp.extract(args)

    litegs.training.start(lp,op,pp,dp,args.test_epochs,args.save_epochs,args.checkpoint_epochs,args.start_checkpoint)