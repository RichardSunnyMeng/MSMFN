import torch
import torch.nn as nn
import torchvision.models.video
from .Resnet import Resnet50_1, Resnet50_2, Resnet50_stem, Resnet50_body
from .Transformer import Transformer, get_transformer_config
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.GlobalPooling = nn.AdaptiveAvgPool2d(1)
        self.RGB_attention = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 3),
            nn.Softmax(-1)
        )

    def forward(self, x):
        channel_weights = self.RGB_attention(self.GlobalPooling(x).view(-1, 3))
        x = torch.mul(x, channel_weights.view(-1, 3, 1, 1))
        return x


class ModalityInteractionAttention(nn.Module):
    def __init__(self, size=(56, 56)):
        super(ModalityInteractionAttention, self).__init__()
        self.size = size
        self.feedforward = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        score_map = self.feedforward(x)  # B * 1 * W * H
        score_map = F.interpolate(score_map, self.size)  # B * 1 * W * H
        B, C, W, H = score_map.shape
        score_map = score_map.view(B, C, -1)
        score_map = self.sigmoid(score_map)
        score_map = score_map.view(B, C, W, H)

        return score_map


class StaticBranch(nn.Module):
    def __init__(self, out_dim=256):
        super(StaticBranch, self).__init__()
        self.CDFI_color = ChannelAttention()
        self.UE_color = ChannelAttention()
        self.ModalityAttention_CDFI = ModalityInteractionAttention()
        self.ModalityAttention_UE = ModalityInteractionAttention()

        self.CDFI_stem = Resnet50_stem()
        self.UE_stem = Resnet50_stem()

        self.CDFI_body = Resnet50_body()
        self.UE_body = Resnet50_body()

        self.US_stage1 = Resnet50_1()
        self.US_stage2 = Resnet50_2()

        self.fc = nn.Sequential(
            nn.Linear(2048, out_dim),
            nn.ReLU()
        )

        self.map_fusion = nn.Sequential(
            nn.Conv2d(512 * 3, 512, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

    def forward(self, x_us, x_cdfi, x_ue):
        # Channel Attention
        x_cdfi = self.CDFI_color(x_cdfi)
        x_ue = self.UE_color(x_ue)

        f_us = self.US_stage1(x_us)  # B * C * W * H
        x_cdfi = self.CDFI_stem(x_cdfi)  # B * C * W * H
        x_ue = self.UE_stem(x_ue)  # B * C * W * H

        # Modality Interaction Attention
        weight_cdfi = self.ModalityAttention_CDFI(f_us)   # B * 1 * W * H
        weight_ue = self.ModalityAttention_UE(f_us)    # B * 1 * W * H

        x_cdfi = torch.mul(x_cdfi, weight_cdfi)
        x_ue = torch.mul(x_ue, weight_ue)

        f_cdfi = self.CDFI_body(x_cdfi)  # B * C * W * H
        f_ue = self.UE_body(x_ue)  # B * C * W * H

        f_map = torch.cat([f_us, f_cdfi, f_ue], dim=1)  # B * C * W * H
        f_map = self.map_fusion(f_map)

        f = self.US_stage2(f_map).squeeze(-1).squeeze(-1)  # B * k
        return self.fc(f)


class DynamicBranch(nn.Module):
    def __init__(self):
        super(DynamicBranch, self).__init__()
        self.r2plus1d = torchvision.models.video.r2plus1d_18()
        self.r2plus1d.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )

    def forward(self, x):   # b * c * t * h * w
        # r2plus1d
        x = self.r2plus1d.stem(x)

        x = self.r2plus1d.layer1(x)
        x = self.r2plus1d.layer2(x)   # B * c * t * 1 * 1

        x = self.r2plus1d.layer3(x)
        x = self.r2plus1d.layer4(x)

        x = self.r2plus1d.avgpool(x)

        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.r2plus1d.fc(x)
        return x


class Model(nn.Module):
    def __init__(self, out_dim=256):
        super(Model, self).__init__()
        self.static_branch = StaticBranch()
        self.dynamic_Branch = DynamicBranch()
        self.transformer = Transformer(get_transformer_config(out_dim))

        self.classificator = nn.Sequential(
            nn.Linear(out_dim, 2),
            nn.Softmax(-1)
        )

        self.AuxiliaryClassificator_dynamic = nn.Sequential(
            nn.Linear(out_dim, 2),
            nn.Softmax(-1)
        )
        self.AuxiliaryClassificator_static = nn.Sequential(
            nn.Linear(out_dim, 2),
            nn.Softmax(-1)
        )

    def forward(self, x_us, x_cdfi, x_ue, x_ceus):
        f_us = self.static_branch(x_us, x_cdfi, x_ue)  # B * k

        x_ceus = x_ceus.permute(0, 2, 1, 3, 4)
        f_ceus = self.dynamic_Branch(x_ceus)  # B * k

        # Transformer融合
        d = torch.cat([f_us.unsqueeze(1), f_ceus.unsqueeze(1)], dim=1)
        d = self.transformer(d)

        return self.classificator(d[:, 0, :]), \
               (f_ceus, self.AuxiliaryClassificator_dynamic(f_ceus)), \
               (f_us, self.AuxiliaryClassificator_static(f_us))


class IndividualModel(nn.Module):
    def __init__(self):
        super(IndividualModel, self).__init__()
        self.img_model = Model()

        self.fc = nn.Sequential(
            nn.Linear(7, 2),
            nn.Softmax(-1)
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(2, 1),
            nn.ReLU()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(2, 1),
            nn.ReLU()
        )
        self.fc_3 = nn.Sequential(
            nn.Linear(6, 1),
            nn.ReLU()
        )

    def parameters_init(self, path):
        state = torch.load(path)
        self.img_model.load_state_dict(state)

    def forward(self, x_us, x_cdfi, x_ue, x_ceus, clinical):
        with torch.no_grad():
            y, _, _ = self.img_model(x_cdfi, x_us, x_ue, x_ceus)
        y_score = y[:, 1:]

        # 0~2 2 2 6
        float_info = clinical[:, 0:3]
        y_1 = self.fc_1(clinical[:, 3:5])
        y_2 = self.fc_2(clinical[:, 5:7])
        y_3 = self.fc_3(clinical[:, 7:])

        y = torch.cat([y_score, float_info, y_1, y_2, y_3], dim=-1)
        o = self.fc(y)
        return o
