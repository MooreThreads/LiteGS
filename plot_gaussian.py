# Importing the necessary modules

import numpy as np

import matplotlib.pyplot as plt


cov = np.array([[2, 0.5], [0.5, 1]])
mean = np.array([0.0,0.0])
cov_inv=np.linalg.inv(cov)
value,vec=np.linalg.eigh(cov)

def gs_power(xy,mean,cov_inv):
    temp=np.matmul(xy,cov_inv)
    temp=temp[0]*xy[0]+temp[1]*xy[1]
    power=np.exp(-0.5*temp)
    return power

# contour of gaussian
alpha=1.0

print("max length:",np.sqrt(value.max()*-2*np.log(1/(255*alpha))))

coord=mean+vec[0]*np.sqrt(value[0]*-2*np.log(1/(255*alpha)))
print(gs_power(coord,mean,cov_inv)*alpha)
print(1.0/255)

coord=mean+vec[1]*np.sqrt(value[1]*-2*np.log(1/(255*alpha)))
print(gs_power(coord,mean,cov_inv)*alpha)
print(1.0/255)

#gaussian split
X=np.arange(-5,5,0.1)
Y=np.arange(-5,5,0.1)
img=np.zeros((X.shape[0],Y.shape[0]))

for index_i,x in enumerate(X):
    for index_j,y in enumerate(Y):
        coord=np.array((x,y))
        img[index_i,index_j]=gs_power(coord,mean,cov_inv)
plt.imshow(img)
plt.show()
pass