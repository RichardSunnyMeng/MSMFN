import torch
from utils import set_seed
from models import Model, IndividualModel
from loss import Loss
import config

set_seed(config.seed)
m = Model()
loss = Loss()

optimizer = torch.optim.Adam(m.parameters(), lr=0.00001, weight_decay=0.000)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

m.train()
x_us = torch.rand(size=(1, 3, config.us_shape[0], config.us_shape[1]))
x_cdfi = torch.rand(size=(1, 3, config.cdfi_shape[0], config.cdfi_shape[1]))
x_ue = torch.rand(size=(1, 3, config.ue_shape[0], config.ue_shape[1]))
x_ceus = torch.rand(size=(1, 10, 3, config.ceus_shape[0], config.ceus_shape[1]))

y, o_dynamic, o_static = m(x_us, x_cdfi, x_ue, x_ceus)
l = loss(y, o_dynamic[1], o_static[1], o_dynamic[0], o_static[0], torch.tensor([1], dtype=torch.int64))
