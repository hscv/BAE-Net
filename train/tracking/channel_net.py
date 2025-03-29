import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn



class Channel_attention_net(nn.Module):

    def __init__(self, channel=16, reduction=4,train=False):
        super(Channel_attention_net, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(channel, channel//2,bias=True),
                                     nn.ReLU(inplace=False),
                                     nn.Linear(channel//2, channel//4,bias=True))

        self.decoder = nn.Sequential(nn.Linear(channel//4, channel//2,bias=True),
                                     nn.ReLU(inplace=False),
                                     nn.Linear(channel//2, channel,bias=True)
                                     # nn.Tanh(),  等会测试数据归一化
                                     )
        self.soft = nn.Softmax(dim=-1)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # if train:
        #     for m in self.modules():
        #         if isinstance(m, nn.Linear):
        #             # torch.nn.init.uniform_(m.weight, 0, 1)
        #             m.weight.data.normal_(1.0, 0.2)
        #             # m.bias.data.zero_()
        #         else :
        #             pass

    def forward(self, x):  # return 16 bands point-mul result
        # print ('x.size() = ',x.size())
        # print ('type(x) = ',type(x))
        b, c, w, h = x.size()# [1,16,127,127]
        c1 = x.view(b,c,-1)
        c2 = c1.permute(0,2,1)
        res1 = self.encoder(c2)
        res2 = self.decoder(res1)

        res2 = res2 / res2.max() # 归一化
        res2 = self.soft(res2)

        res = res2.permute(0,2,1)
        att = res.view(b,c,w,h)
        y = res.mean(dim=2)
        orderY = torch.sort(y, dim=-1, descending=True, out=None)  # 排序
        y = orderY[0]  # 排序的方向求导机制是如何的，会对y造成影响吗
        # print ('orderY = ',orderY)
        y = y.view(b, c, 1, 1) # [1,16,1,1]
        att = y.expand_as(x) # [1,16,127,127]
        res = x.mul(att) # [1,16,127,127]
        res = x
        return res,orderY
