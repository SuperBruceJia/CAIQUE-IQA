# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

from model.layers import TransformerBlock, OverlapPatchEmbed, L2pooling, LayerNorm, Downsample
from model.layers import SPPLayer, init_linear
from lib.utils import sum_multiply, bilinear_pool, gaussian_prior


class Network(nn.Module):
    def __init__(self,
                 args,
                 dim=[16, 32, 64, 128, 256, 512],
                 num_blocks=[1, 1, 2, 2, 3, 3],
                 heads=[8, 8, 8, 8, 8, 8],
                 bias=True,
                 LayerNorm='WithBias',
                 ratio=[1, 1, 1, 1, 1, 1]
                 ):

        super(Network, self).__init__()

        self.num_dis_type = args.num_dis_type

        # Pre-trained VGGNet 19
        self.cnn_backbone = models.vgg19(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()

        for x in range(0, 5):
            self.stage1.add_module(str(x), self.cnn_backbone._modules[str(x)])

        for x in range(5, 10):
            self.stage2.add_module(str(x), self.cnn_backbone._modules[str(x)])

        for x in range(10, 19):
            self.stage3.add_module(str(x), self.cnn_backbone._modules[str(x)])

        for x in range(19, 28):
            self.stage4.add_module(str(x), self.cnn_backbone._modules[str(x)])

        for x in range(28, 37):
            self.stage5.add_module(str(x), self.cnn_backbone._modules[str(x)])

        self.refine_1_0 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.refine_2_0 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.refine_3_0 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.refine_4_0 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.refine_5_0 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=True)

        self.refine_1_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True)
        self.refine_2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True)
        self.refine_3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True)
        self.refine_4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True)
        self.refine_5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True)

        self.refine_1_2 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=True)
        self.refine_2_2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.refine_3_2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.refine_4_2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.refine_5_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)

        self.spp1 = SPPLayer(8)
        self.spp2 = SPPLayer(4)
        self.spp3 = SPPLayer(4)
        self.spp4 = SPPLayer(2)
        self.spp5 = SPPLayer(1)

        #########################################################################################################
        # Non-local Modeling
        self.encoder_0 = nn.Sequential(*[TransformerBlock(dim=dim[0],
                                                          num_heads=heads[0],
                                                          bias=bias,
                                                          LayerNormType=LayerNorm,
                                                          ratio=ratio[0])
                                         for i in range(num_blocks[0])])

        self.encoder_1 = nn.Sequential(*[TransformerBlock(dim=dim[1],
                                                          num_heads=heads[1],
                                                          bias=bias,
                                                          LayerNormType=LayerNorm,
                                                          ratio=ratio[1])
                                         for i in range(num_blocks[1])])

        self.encoder_2 = nn.Sequential(*[TransformerBlock(dim=dim[2],
                                                          num_heads=heads[2],
                                                          bias=bias,
                                                          LayerNormType=LayerNorm,
                                                          ratio=ratio[2])
                                         for i in range(num_blocks[2])])

        self.encoder_3 = nn.Sequential(*[TransformerBlock(dim=dim[3],
                                                          num_heads=heads[3],
                                                          bias=bias,
                                                          LayerNormType=LayerNorm,
                                                          ratio=ratio[3])
                                         for i in range(num_blocks[3])])

        self.encoder_4 = nn.Sequential(*[TransformerBlock(dim=dim[4],
                                                          num_heads=heads[4],
                                                          bias=bias,
                                                          LayerNormType=LayerNorm,
                                                          ratio=ratio[4])
                                         for i in range(num_blocks[4])])

        self.encoder_5 = nn.Sequential(*[TransformerBlock(dim=dim[5],
                                                          num_heads=heads[5],
                                                          bias=bias,
                                                          LayerNormType=LayerNorm,
                                                          ratio=ratio[5])
                                         for i in range(num_blocks[5])])

        #########################################################################################################
        self.patch_embed = OverlapPatchEmbed(3, 16)
        self.down0_1 = Downsample(n_feat=dim[0])
        self.down1_2 = Downsample(n_feat=dim[1])
        self.down2_3 = Downsample(n_feat=dim[2])
        self.down3_4 = Downsample(n_feat=dim[3])
        self.down4_5 = Downsample(n_feat=dim[4])

        self.nspp0 = SPPLayer(8)
        self.nspp1 = SPPLayer(4)
        self.nspp2 = SPPLayer(4)
        self.nspp3 = SPPLayer(2)
        self.nspp4 = SPPLayer(2)
        self.nspp5 = SPPLayer(1)

        #########################################################################################################
        # FC For Quality Score Regression - Non-local
        self.quality_linear1 = nn.Linear((16 * sum_multiply(8)
                                          + 32 * sum_multiply(4)
                                          + 64 * sum_multiply(4)
                                          + 128 * sum_multiply(2)
                                          + 256 * sum_multiply(2)
                                          + 512 * sum_multiply(1)) * 4,
                                         128)
        self.quality_linear2 = nn.Linear(128, 64)
        self.quality_linear3 = nn.Linear(64, 1)

        # FC For Quality Score Regression - Local
        self.quality_linear4 = nn.Linear((16 * sum_multiply(8)
                                          + 32 * sum_multiply(4)
                                          + 64 * sum_multiply(4)
                                          + 128 * sum_multiply(2)
                                          + 256 * sum_multiply(1)) * 4,
                                         128)
        self.quality_linear5 = nn.Linear(128, 64)
        self.quality_linear6 = nn.Linear(64, 1)

        # FC For Quality Score Regression - Local and Non-local Fusion
        # self.quality_linear7 = nn.Linear(16 * 16 + 32 * 32 + 64 * 64 + 128 * 128 + 256 * 256, 512)
        self.quality_linear7 = nn.Linear((16 * sum_multiply(8)
                                           + 32 * sum_multiply(4)
                                           + 64 * sum_multiply(4)
                                           + 128 * sum_multiply(2)
                                           + 256 * sum_multiply(2)
                                           + 512 * sum_multiply(1)) * 4
                                          + (16 * sum_multiply(8)
                                             + 32 * sum_multiply(4)
                                             + 64 * sum_multiply(4)
                                             + 128 * sum_multiply(2)
                                             + 256 * sum_multiply(1)) * 4,
                                          128)
        self.quality_linear8 = nn.Linear(128, 64)
        self.quality_linear9 = nn.Linear(64, 1)

        # # FC For Distortion Type Classification
        # self.dis_type_linear1 = nn.Linear((8 * sum_multiply(4)
        #                                    + 16 * sum_multiply(4)
        #                                    + 32 * sum_multiply(2)
        #                                    + 64 * sum_multiply(2)
        #                                    + 128 * sum_multiply(1)
        #                                    + 256 * sum_multiply(1)) * 4
        #                                   + (16 * sum_multiply(4)
        #                                      + 32 * sum_multiply(4)
        #                                      + 64 * sum_multiply(2)
        #                                      + 128 * sum_multiply(2)
        #                                      + 256 * sum_multiply(1)) * 4, 512)
        # # self.dis_type_linear2 = nn.Linear(512, 64)
        # self.dis_type_linear3 = nn.Linear(512, self.num_dis_type)

    def forward(self, patch):
        local_feat1 = self.stage1(patch)
        local_feat2 = self.stage2(local_feat1)
        local_feat3 = self.stage3(local_feat2)
        local_feat4 = self.stage4(local_feat3)
        local_feat5 = self.stage5(local_feat4)

        local_feat1 = F.elu(self.refine_1_0(local_feat1))
        local_feat1 = F.elu(self.refine_1_1(local_feat1))
        local_feat1 = F.elu(self.refine_1_2(local_feat1))
        lq_1 = self.spp1(local_feat1)

        local_feat2 = F.elu(self.refine_2_0(local_feat2))
        local_feat2 = F.elu(self.refine_2_1(local_feat2))
        local_feat2 = F.elu(self.refine_2_2(local_feat2))
        lq_2 = self.spp2(local_feat2)

        local_feat3 = F.elu(self.refine_3_0(local_feat3))
        local_feat3 = F.elu(self.refine_3_1(local_feat3))
        local_feat3 = F.elu(self.refine_3_2(local_feat3))
        lq_3 = self.spp3(local_feat3)
        
        local_feat4 = F.elu(self.refine_4_0(local_feat4))
        local_feat4 = F.elu(self.refine_4_1(local_feat4))
        local_feat4 = F.elu(self.refine_4_2(local_feat4))
        lq_4 = self.spp4(local_feat4)

        local_feat5 = F.elu(self.refine_5_0(local_feat5))
        local_feat5 = F.elu(self.refine_5_1(local_feat5))
        local_feat5 = F.elu(self.refine_5_2(local_feat5))
        lq_5 = self.spp5(local_feat5)

        feat_l = torch.cat([lq_1, lq_2, lq_3, lq_4, lq_5], dim=1)

        # input: torch.Size([4, 3, 224, 224])
        embedding = self.patch_embed(patch)  # torch.Size([4, 8, 224, 224])
        non_feat0 = self.encoder_0(embedding)  # torch.Size([4, 8, 224, 224])
        nlq_0 = self.nspp0(non_feat0)

        # feat1: torch.Size([4, 8, 112, 112])
        non_feat1_down = self.down0_1(non_feat0)  # torch.Size([4, 16, 112, 112])
        # non_feat1 = F.elu(local_feat1 + non_feat1)
        # local_feat1: 16
        non_feat1 = torch.cat([local_feat1, non_feat1_down], dim=1)
        non_feat1_encode = self.encoder_1(non_feat1)  # torch.Size([4, 32, 112, 112])
        nlq_1 = self.nspp1(non_feat1_encode)

        # feat2: torch.Size([4, 16, 56, 56])
        non_feat2_down = self.down1_2(non_feat1_encode)  # torch.Size([4, 32, 56, 56])
        # non_feat2 = F.elu(local_feat2 + non_feat2)
        # local_feat2: 32
        non_feat2 = torch.cat([local_feat2, non_feat2_down], dim=1)
        non_feat2_encode = self.encoder_2(non_feat2)  # torch.Size([4, 64, 56, 56])
        nlq_2 = self.nspp2(non_feat2_encode)

        # feat3: torch.Size([4, 32, 28, 28])
        non_feat3_down = self.down2_3(non_feat2_encode)  # torch.Size([4, 64, 28, 28])
        # non_feat3 = F.elu(local_feat3 + non_feat3)
        # local_feat3: 64
        non_feat3 = torch.cat([local_feat3, non_feat3_down], dim=1)
        non_feat3_encode = self.encoder_3(non_feat3)  # torch.Size([4, 128, 28, 28])
        nlq_3 = self.nspp3(non_feat3_encode)

        # feat4: torch.Size([4, 64, 14, 14])
        non_feat4_down = self.down3_4(non_feat3_encode)  # torch.Size([4, 128, 14, 14])
        # non_feat4 = F.elu(local_feat4 + non_feat4)
        # local_feat4: 128
        non_feat4 = torch.cat([local_feat4, non_feat4_down], dim=1)
        non_feat4_encode = self.encoder_4(non_feat4)  # torch.Size([4, 256, 14, 14])
        nlq_4 = self.nspp4(non_feat4_encode)

        # feat5: torch.Size([4, 128, 7, 7])
        non_feat5_down = self.down4_5(non_feat4_encode)  # torch.Size([4, 256, 7, 7])
        # non_feat5 = F.elu(local_feat5 + non_feat5)
        # local_feat5: 256
        non_feat5 = torch.cat([local_feat5, non_feat5_down], dim=1)
        non_feat5_encode = self.encoder_5(non_feat5)  # torch.Size([4, 512, 7, 7])
        nlq_5 = self.nspp5(non_feat5_encode)

        feat_nl = torch.cat([nlq_0, nlq_1, nlq_2, nlq_3, nlq_4, nlq_5], dim=1)

        # # Bilinear Features
        # bilinear_feat_1 = bilinear_pool(feature_1=non_feat1_down, feature_2=local_feat1)
        # bilinear_feat_2 = bilinear_pool(feature_1=non_feat2_down, feature_2=local_feat2)
        # bilinear_feat_3 = bilinear_pool(feature_1=non_feat3_down, feature_2=local_feat3)
        # bilinear_feat_4 = bilinear_pool(feature_1=non_feat4_down, feature_2=local_feat4)
        # bilinear_feat_5 = bilinear_pool(feature_1=non_feat5_down, feature_2=local_feat5)
        # overall_feat = torch.cat([bilinear_feat_1, bilinear_feat_2,
        #                          bilinear_feat_3, bilinear_feat_4,
        #                          bilinear_feat_5], dim=1)
        overall_feat = torch.cat([feat_nl, feat_l], dim=1)

        # For Quality Prediction - Non-local Features
        quality_nl_out = F.elu(self.quality_linear1(feat_nl))
        quality_nl_out = F.elu(self.quality_linear2(quality_nl_out))
        quality_nl_out = self.quality_linear3(quality_nl_out)

        # For Quality Prediction - Local Features
        quality_l_out = F.elu(self.quality_linear4(feat_l))
        quality_l_out = F.elu(self.quality_linear5(quality_l_out))
        quality_l_out = self.quality_linear6(quality_l_out)

        # For Quality Prediction - Non-local and Local Fusion Features
        quality_fusion_out = F.elu(self.quality_linear7(overall_feat))
        quality_fusion_out = F.elu(self.quality_linear8(quality_fusion_out))
        quality_fusion_out = self.quality_linear9(quality_fusion_out)

        # # For Distortion Type Classification
        # dis_type_out = F.elu(self.dis_type_linear1(overall_feat))
        # # # dis_type_out = F.elu(self.dis_type_linear2(dis_type_out))
        # dis_type_out = F.softmax(self.dis_type_linear3(dis_type_out), dim=1)

        # return quality_fusion_out, quality_nl_out, quality_l_out, dis_type_out
        return quality_fusion_out, quality_nl_out, quality_l_out
