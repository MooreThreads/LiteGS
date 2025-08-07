import torch;import torch_musa
import litegs.utils
# import matplotlib.pyplot as plt
import time

datas=torch.load("./profiler_input_data/raster_input.pth",map_location=torch.device('musa'),weights_only=False)
img,transmitance,depth,normal=litegs.utils.wrapper.GaussiansRasterFunc.apply(*datas)
img=litegs.utils.tiles2img_torch(img,82,105)[...,:840,:1297].contiguous()
img.mean().backward()
torch.musa.synchronize()
#plt.imsave("./render.png",img[0].permute(1,2,0).detach().cpu())

# render_tiles=[0]
# # for y in range(64):
# #     for x in range(64):
# #         render_tiles.append(y*82+x)
# render_tiles=torch.tensor(render_tiles,dtype=torch.int32).unsqueeze(0).cuda()+1
# datas[6]=render_tiles
loop=1000
start_time=time.time()
for i in range(loop):
    img,transmitance,depth,normal=litegs.utils.wrapper.GaussiansRasterFunc.apply(*datas)
    img=litegs.utils.tiles2img_torch(img,82,105)[...,:840,:1297].contiguous()
    img.mean().backward()
torch.musa.synchronize()
end_time=time.time()
print("takes:{}ms".format((end_time-start_time)/loop*1000))