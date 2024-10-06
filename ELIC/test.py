# 创建 TestModel 实例并加载权重
from Network import TestModel
import torch
from Transformer_Network_ga import Ga_Net, TS_Net
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'


checkpoints = "checkpoint/ELIC_arch_base/mse_0.045/checkpoint_best_loss.pth.tar"

test_model = TestModel()
test_model.load_state_dict(torch.load(checkpoints))

# 创建 Ga_Net 实例
# ga_net = Ga_Net()
net = TS_Net()

# 获取 TestModel 和 Ga_Net 的状态字典
test_model_state_dict = test_model.state_dict()
ga_net_state_dict = net.state_dict()

# 将 TestModel 中与 Ga_Net 一致的部分的权重赋值给 Ga_Net
# 将 TestModel 中的 self.ga 的权重分别加载到 Ga_Net 中对应的子集中

for k, v in ga_net_state_dict.items():
    # print("k1---net", k)
    ga_net_sub_module_key = f'Teacher_Net.g_a.{0}.'
    if k.startswith(ga_net_sub_module_key):
        print("first teacher -- V ===", v)



for k, v in test_model_state_dict.items():
    test_model_sub_module_key = f'g_a.{0}.'
    if k.startswith(test_model_sub_module_key):
        print("k2-test --net", k)
        print("test -- V ===", v)

for i in range(0, 15):
    test_model_sub_module_key = f'g_a.{i}.'
    ga_net_sub_module_key = f'Teacher_Net.g_a.{i}.'
    for k, v in test_model_state_dict.items():
        if k.startswith(test_model_sub_module_key):
            print("k == ", k)
            new_key = k.replace(test_model_sub_module_key, ga_net_sub_module_key)
            ga_net_state_dict[new_key].copy_(v)


for k, v in ga_net_state_dict.items():
    # print("k1---net", k)
    ga_net_sub_module_key = f'Teacher_Net.g_a.{0}.'
    if k.startswith(ga_net_sub_module_key):
        print("second teacher -- V ===", v)