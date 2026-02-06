import torch;import torch_musa
import litegs.utils
import time

#args=torch.load('./profiler_input_data/crossroad_raster_input.pt')
args=torch.load('./profiler_input_data/garden_raster_input.pt')

for i in range(10):
    img,transmitance,depth,normal,lst_contributor=litegs.utils.wrapper.GaussiansRasterFunc.apply(*args)
    img.mean().backward()

torch.musa.synchronize()

loops_num=100
start=time.time()
for i in range(loops_num):
    img,transmitance,depth,normal,lst_contributor=litegs.utils.wrapper.GaussiansRasterFunc.apply(*args)
    img.mean().backward()
torch.musa.synchronize()
end=time.time()
print('gaussian raster forward&backward: ', (end-start)*1000/loops_num, 'ms')