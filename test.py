import torch
import litegs.utils
import matplotlib.pyplot as plt
import time

datas=torch.load("./profiler_input_data/raster_input.pth",map_location=torch.device('musa'),weights_only=False)


img,transmitance,depth,normal=litegs.utils.wrapper.GaussiansRasterFunc.apply(*datas)
img=litegs.utils.tiles2img_torch(img,82,105)[...,:840,:1297].contiguous()
img.mean().backward()
torch.musa.synchronize()

loop=100
start_time=time.time()
for i in range(loop):
    img,transmitance,depth,normal=litegs.utils.wrapper.GaussiansRasterFunc.apply(*datas)
    img=litegs.utils.tiles2img_torch(img,82,105)[...,:840,:1297].contiguous()
    img.mean().backward()
torch.musa.synchronize()
end_time=time.time()
print("takes:{}ms".format((end_time-start_time)/loop*1000))