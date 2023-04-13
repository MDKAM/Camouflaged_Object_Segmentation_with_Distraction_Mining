import torch
import torch.nn as nn
import torch.nn.functional as F

import backbone.resnet.resnet as resnet

# ################## Positioning Module ######################
class PositioningM(nn.Module):
    def __init__(self, channel):
        super(PositioningM, self).__init__()
        self.channel = channel
        self.gamma_ca = nn.Parameter(torch.ones(1))
        self.gamma_sa = nn.Parameter(torch.ones(1))
        self.softmax= nn.Softmax(dim=-1)
        self.query_conv = nn.Conv2d(in_channels=channel, out_channels=channel // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=channel, out_channels=channel // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.map = nn.Conv2d(channel, 1, 7, 1, 3)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        batch, C, H, W = x.size()
        
        # Channel attention block
        proj_query_ca = x.view(batch, C, -1)
        proj_key_ca = x.view(batch, C, -1).permute(0, 2, 1)
        score_ca = torch.bmm(proj_query_ca, proj_key_ca)
        attention_ca = self.softmax(score_ca)
        proj_value_ca = x.view(batch, C, -1)
        cao = torch.bmm(attention_ca, proj_value_ca)
        cao = cao.view(batch, C, H, W)
        cao = self.gamma_ca * cao + x

        # Spatial attention block
        proj_query_sa = self.query_conv(cao).view(batch, -1, W * H).permute(0, 2, 1)
        proj_key_sa = self.key_conv(cao).view(batch, -1, W * H)
        score_sa = torch.bmm(proj_query_sa, proj_key_sa)
        attention_sa = self.softmax(score_sa)
        proj_value_sa = self.value_conv(cao).view(batch, -1, W * H)
        sao = torch.bmm(proj_value_sa, attention_sa.permute(0, 2, 1))
        sao = sao.view(batch, C, H, W)
        sao = self.gamma_sa * sao + cao
        
        # Map layer
        map = self.map(sao)

        return sao, map


# ################## Context Exploration Block ####################
class Context_Exploration_Block(nn.Module):
    def __init__(self, in_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = in_channels
        self.channels_r = int(in_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_r, 1, 1, 0),
            nn.BatchNorm2d(self.channels_r), nn.ReLU())
        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_r, self.channels_r, 1, 1, 0),
            nn.BatchNorm2d(self.channels_r), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_r, self.channels_r, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_r), nn.ReLU())
        

        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_r, 1, 1, 0),
            nn.BatchNorm2d(self.channels_r), nn.ReLU())
        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_r, self.channels_r, 3, 1, 1),
            nn.BatchNorm2d(self.channels_r), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_r, self.channels_r, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_r), nn.ReLU())
        

        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_r, 1, 1, 0),
            nn.BatchNorm2d(self.channels_r), nn.ReLU())
        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_r, self.channels_r, 5, 1, 2),
            nn.BatchNorm2d(self.channels_r), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_r, self.channels_r, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_r), nn.ReLU())
        

        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_r, 1, 1, 0),
            nn.BatchNorm2d(self.channels_r), nn.ReLU())
        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_r, self.channels_r, 7, 1, 3),
            nn.BatchNorm2d(self.channels_r), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_r, self.channels_r, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_r), nn.ReLU())

        self.fusion = nn.Sequential(
            nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
            nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce

# ######################## Focus Module ###########################
class Focus(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

        self.up = nn.Sequential(
            nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
            nn.BatchNorm2d(self.channel1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self.input_map = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2), 
            nn.Sigmoid())
        
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)

        self.fp = Context_Exploration_Block(self.channel1)
        self.fn = Context_Exploration_Block(self.channel1)

        self.bnRelu1 = nn.Sequential(
            nn.BatchNorm2d(self.channel1),
            nn.ReLU())
        self.bnRelu2 = nn.Sequential(
            nn.BatchNorm2d(self.channel1),
            nn.ReLU())

    def forward(self, x, y, in_map):
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction

        input_map = self.input_map(in_map)
        f_attention = x * input_map
        b_attention = x * (1 - input_map)

        fpd = self.fp(f_attention)                      #false-positive distractions
        fnd = self.fn(b_attention)                      #false-negative distractions

        up = self.up(y)

        refine1 = up - (self.alpha * fpd)
        refine1 = self.bnRelu1(refine1)

        refine2 = refine1 + (self.beta * fnd)
        refine2 = self.bnRelu2(refine2)

        output_map = self.output_map(refine2)

        return refine2, output_map

# ########################## NETWORK ##############################
class PFNet(nn.Module):
    def __init__(self, backbone_path=None):
        super(PFNet, self).__init__()
        # params

        # backbone
        resnet50 = resnet.resnet50(backbone_path)
        self.layer0 = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu)
        self.layer1 = nn.Sequential(resnet50.maxpool, resnet50.layer1)
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        # channel reduction
        self.cr4 = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.cr1 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())

        # positioning
        self.positioning = PositioningM(512)

        # focus
        self.focus3 = Focus(256, 512)
        self.focus2 = Focus(128, 256)
        self.focus1 = Focus(64, 128)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        # x: [batch_size, channel=3, h, w]
        layer0 = self.layer0(x)                                 # [-1, 64, h/2, w/2]
        layer1 = self.layer1(layer0)                            # [-1, 256, h/4, w/4]
        layer2 = self.layer2(layer1)                            # [-1, 512, h/8, w/8]
        layer3 = self.layer3(layer2)                            # [-1, 1024, h/16, w/16]
        layer4 = self.layer4(layer3)                            # [-1, 2048, h/32, w/32]

        # channel reduction
        f4 = self.cr4(layer4)
        f3 = self.cr3(layer3)
        f2 = self.cr2(layer2)
        f1 = self.cr1(layer1)

        # positioning
        positioning, predict4 = self.positioning(f4)

        # focus
        focus3, predict3 = self.focus3(f3, positioning, predict4)
        focus2, predict2 = self.focus2(f2, focus3, predict3)
        focus1, predict1 = self.focus1(f1, focus2, predict2)

        # rescale to original size
        predictions = [predict4, predict3, predict2, predict1]

        for i in range(len(predictions)):
            predictions[i] = F.interpolate(predictions[i], size=x.size()[2:], mode='bilinear', align_corners=True)

        predict4, predict3, predict2, predict1 = predictions


        if self.training:
            return predict4, predict3, predict2, predict1

        return torch.sigmoid(predict4), torch.sigmoid(predict3), torch.sigmoid(predict2), torch.sigmoid(
            predict1)
