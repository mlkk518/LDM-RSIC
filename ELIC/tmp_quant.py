##  mlkk 测试 对舍弃的通道的
import numpy  as np
import torch


y = torch.randn(4, 8, 4, 4)

y1 = y.clone()
y1[:,0:3, :,:] = 0

y2 = y.clone()

##  根据y1 中 将y2中对应的通道置零
B, C, H, W = y1.size()
y1 = torch.reshape(y1, (B, C, -1))
sum_t = torch.reshape(abs(torch.sign(torch.sum(y1, dim=2))), (B, C, 1, 1))
yy = y2*sum_t

print("y shape ", y.shape)
# print("y shape", y.shape)
print("y1 shape", y1.shape)

print("sum_t", sum_t.shape)


print("y1  $$$$$  ", y1)
print("y2  $$$$$  ", y2)
print("yy  $$$$$$$$$$$$   ", yy)
