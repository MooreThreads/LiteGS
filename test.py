import torch
import litegs.utils
import matplotlib.pyplot as plt
import time

datas=torch.load("./profiler_input_data/raster_input.pth",map_location=torch.device('musa'),weights_only=False)

visible_primitives_num=datas[1][0,2:]-datas[1][0,1:-1]
sorted_tileid=visible_primitives_num.sort(descending=True)[1].int().reshape(1,-1)+1
#datas[6]=sorted_tileid

tiles_0=(visible_primitives_num<256).nonzero().int().reshape(1,-1)
tiles_1=((256<=visible_primitives_num)&(visible_primitives_num<512)).nonzero().int().reshape(1,-1)
tiles_2=((512<=visible_primitives_num)&(visible_primitives_num<1024)).nonzero().int().reshape(1,-1)
tiles_3=((1024<=visible_primitives_num)&(visible_primitives_num<2048)).nonzero().int().reshape(1,-1)
tiles_4=((2048<=visible_primitives_num)&(visible_primitives_num<4096)).nonzero().int().reshape(1,-1)
tiles_5=((4096<=visible_primitives_num)).nonzero().int().reshape(1,-1)
#datas[6]=torch.cat([tiles_5,tiles_4,tiles_3,tiles_2,tiles_1,tiles_0],dim=1).contiguous()+1



# img,transmitance,depth,normal=litegs.utils.wrapper.GaussiansRasterFunc.apply(*datas)
# img=litegs.utils.tiles2img_torch(img,82,105)[...,:840,:1297].contiguous()
# img.mean().backward()
#plt.imsave("./render.png",img[0].permute(1,2,0).detach().cpu())

loop=10
start_time=time.time()
for i in range(loop):
    img,transmitance,depth,normal=litegs.utils.wrapper.GaussiansRasterFunc.apply(*datas)
    img=litegs.utils.tiles2img_torch(img,82,105)[...,:840,:1297].contiguous()
    torch.musa.synchronize()
    img.mean().backward()
    torch.musa.synchronize()
end_time=time.time()
print("takes:{}ms".format((end_time-start_time)/loop*1000))