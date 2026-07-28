""" Code adapted from https://github.com/hassony2/inflated_convnets_pytorch """

import math
import ipdb
import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
from merlin.models import inflate


class I3ResNet(torch.nn.Module):
    def __init__(
        self, resnet2d, frame_nb=16, class_nb=1000, conv_class=False, return_skips=False, ImageEmbedding=False
    ):
        """
        Args:
            conv_class: Whether to use convolutional layer as classifier to
                adapt to various number of frames
        """
        super(I3ResNet, self).__init__()
        self.return_skips = return_skips
        self.conv_class = conv_class
        self.ImageEmbedding = ImageEmbedding

        self.conv1 = inflate.inflate_conv(
            resnet2d.conv1, time_dim=3, time_padding=1, center=True
        )
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(
            resnet2d.maxpool, time_dim=3, time_padding=1, time_stride=2
        )

        self.layer1 = inflate_reslayer(resnet2d.layer1)
        self.layer2 = inflate_reslayer(resnet2d.layer2)
        self.layer3 = inflate_reslayer(resnet2d.layer3)
        self.layer4 = inflate_reslayer(resnet2d.layer4)

        # if conv_class:
        #     self.avgpool = inflate.inflate_pool(resnet2d.avgpool, time_dim=1)
        #     self.classifier = torch.nn.Conv3d(
        #         in_channels=2048,
        #         out_channels=class_nb,
        #         kernel_size=(1, 1, 1),
        #         bias=True,
        #     )
        #     self.contrastive_head = torch.nn.Conv3d(
        #         in_channels=2048, out_channels=512, kernel_size=(1, 1, 1), bias=True
        #     )
        # else:
        #     final_time_dim = int(math.ceil(frame_nb / 16))
        #     self.avgpool = inflate.inflate_pool(
        #         resnet2d.avgpool, time_dim=final_time_dim
        #     )
        #     self.fc = inflate.inflate_linear(resnet2d.fc, 1)

    def forward(self, x):
        skips = []
        # ipdb.set_trace()
        x = x.permute(0, 1, 4, 2, 3)
        x = torch.cat((x, x, x), dim=1)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.maxpool(x)

        # # 删掉了checkpoint
        # # x = checkpoint.checkpoint(self.layer1, x)
        # x = self.layer1(x)
        # # ipdb.set_trace()
        # if self.return_skips:
        #     skips.append(x.permute(0, 1, 3, 4, 2))
        # # x = checkpoint.checkpoint(self.layer2, x)
        # x = self.layer2(x)
        # # ipdb.set_trace()
        # if self.return_skips:
        #     skips.append(x.permute(0, 1, 3, 4, 2))
        # # x = checkpoint.checkpoint(self.layer3, x)
        # x = self.layer3(x)
        # # ipdb.set_trace()
        # if self.return_skips:
        #     skips.append(x.permute(0, 1, 3, 4, 2))
        # # x = checkpoint.checkpoint(self.layer4, x)
        # x = self.layer4(x)
        # # ipdb.set_trace()
        # if self.return_skips:
        #     skips.append(x.permute(0, 1, 3, 4, 2))

        x = checkpoint.checkpoint(self.layer1, x,use_reentrant=False)
        # ipdb.set_trace()
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer2, x,use_reentrant=False)
        # ipdb.set_trace()
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer3, x,use_reentrant=False)
        # ipdb.set_trace()
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer4, x,use_reentrant=False)
        # ipdb.set_trace()
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))

        # ipdb.set_trace()
        return x
                    
        if self.conv_class:
            x_features = self.avgpool(x)
            
            if self.ImageEmbedding:
                return x_features.squeeze(2).squeeze(2).squeeze(2).unsqueeze(0)
            
            x_ehr = self.classifier(x_features)
            x_ehr = x_ehr.squeeze(3)
            x_ehr = x_ehr.squeeze(3)
            x_ehr = x_ehr.mean(2)
            x_contrastive = self.contrastive_head(x_features)
            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.mean(2)
            if self.return_skips:
                return x_contrastive, x_ehr, skips
            else:
                return x_contrastive, x_ehr
        else:
            x = self.avgpool(x)
            x_reshape = x.view(x.size(0), -1)
            x = self.fc(x_reshape)
        return x

class I3ResNet_modified(nn.Module):
    def __init__(self, block, layers, frame_nb=16, class_nb=1000, conv_class=False, return_skips=False, ImageEmbedding=False):
        """
        直接构建的 Inflated 3D ResNet (I3D)。

        Args:
            block (nn.Module): 要使用的Block类型 (e.g., Bottleneck3d)。
            layers (list[int]): 每个ResNet stage中的block数量 (e.g., [3, 4, 6, 3] for I3D-50)。
            class_nb (int): 输出类别的数量。
            ... 其他参数 ...
        """
        super(I3ResNet_modified, self).__init__()
        self.return_skips = return_skips
        self.conv_class = conv_class
        self.ImageEmbedding = ImageEmbedding
        
        self.inplanes = 64 # ResNet的初始通道数

        # Stage 1: Conv1 + MaxPool
        # 对应 inflate.inflate_conv(resnet2d.conv1, time_dim=3, ...)
        # time_kernel=3, time_stride=1, time_padding=1
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=False)
        # 对应 inflate.inflate_pool(resnet2d.maxpool, time_dim=3, ...)
        # time_kernel=3, time_stride=2, time_padding=1
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        # ResNet Stages
        self.layer1 = self._make_layer(block, 64, layers[0], time_stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, time_stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, time_stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, time_stride=2)

        # # Classifier
        # if conv_class:
        #     self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1)) # 使用AdaptiveAvgPool3d更灵活
        #     self.classifier = nn.Conv3d(
        #         in_channels=512 * block.expansion,
        #         out_channels=class_nb,
        #         kernel_size=(1, 1, 1),
        #         bias=True,
        #     )
        #     # 你可以保留这个，如果需要的话
        #     self.contrastive_head = nn.Conv3d(
        #         in_channels=512 * block.expansion, 
        #         out_channels=512, 
        #         kernel_size=(1, 1, 1), 
        #         bias=True
        #     )
        # else:
        #     # 非卷积分类器的逻辑可以保持，但AdaptiveAvgPool3d通常更好
        #     self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        #     self.fc = nn.Linear(512 * block.expansion, class_nb)
        
        # 初始化权重 (非常重要！)
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1, time_stride=1):
        downsample = None
        # 当空间步长或输入/输出通道数不匹配时，需要downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, 
                    stride=(time_stride, stride, stride), # Downsample也需要时间步长
                    bias=False
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        # 第一个block处理downsample和stride
        layers.append(block(self.inplanes, planes, stride, downsample, time_stride))
        
        self.inplanes = planes * block.expansion # 更新inplanes为下一个stage做准备
        
        # 剩下的blocks
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 你的forward逻辑可以保持不变，因为它处理的是数据流，与模型定义无关
        # ... (复制你原来的 forward 代码到这里) ...
        skips = []
        # ipdb.set_trace()
        # x = x.permute(0, 1, 4, 2, 3)
        # x = torch.cat((x, x, x), dim=1)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.maxpool(x)

        # # 删掉了checkpoint
        # # x = checkpoint.checkpoint(self.layer1, x)
        # x = self.layer1(x)
        # # ipdb.set_trace()
        # if self.return_skips:
        #     skips.append(x.permute(0, 1, 3, 4, 2))
        # # x = checkpoint.checkpoint(self.layer2, x)
        # x = self.layer2(x)
        # # ipdb.set_trace()
        # if self.return_skips:
        #     skips.append(x.permute(0, 1, 3, 4, 2))
        # # x = checkpoint.checkpoint(self.layer3, x)
        # x = self.layer3(x)
        # # ipdb.set_trace()
        # if self.return_skips:
        #     skips.append(x.permute(0, 1, 3, 4, 2))
        # # x = checkpoint.checkpoint(self.layer4, x)
        # x = self.layer4(x)
        # # ipdb.set_trace()
        # if self.return_skips:
        #     skips.append(x.permute(0, 1, 3, 4, 2))

        # 使用梯度检查点 (gradient checkpointing)
        x = torch.utils.checkpoint.checkpoint(self.layer1, x, use_reentrant=False)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = torch.utils.checkpoint.checkpoint(self.layer2, x, use_reentrant=False)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = torch.utils.checkpoint.checkpoint(self.layer3, x, use_reentrant=False)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = torch.utils.checkpoint.checkpoint(self.layer4, x, use_reentrant=False)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))

        # 根据你的需要返回
        # 你的原始代码中 I3ResNet 的 forward 在 layer4 之后就返回了
        # 如果需要完整的分类，需要添加 avgpool 和 classifier 的部分
        return x

class I3ResNet_classify(torch.nn.Module):
    def __init__(
        self, resnet2d, frame_nb=16, class_nb=1000, conv_class=False, return_skips=False, ImageEmbedding=False
    ):
        """
        Args:
            conv_class: Whether to use convolutional layer as classifier to
                adapt to various number of frames
        """
        super(I3ResNet_classify, self).__init__()
        self.return_skips = return_skips
        self.conv_class = conv_class
        self.ImageEmbedding = ImageEmbedding

        self.conv1 = inflate.inflate_conv(
            resnet2d.conv1, time_dim=3, time_padding=1, center=True
        )
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(
            resnet2d.maxpool, time_dim=3, time_padding=1, time_stride=2
        )

        self.layer1 = inflate_reslayer(resnet2d.layer1)
        self.layer2 = inflate_reslayer(resnet2d.layer2)
        self.layer3 = inflate_reslayer(resnet2d.layer3)
        self.layer4 = inflate_reslayer(resnet2d.layer4)

        if conv_class:
            self.avgpool = inflate.inflate_pool(resnet2d.avgpool, time_dim=1)
            # self.classifier = torch.nn.Conv3d(
            #     in_channels=2048,
            #     out_channels=class_nb,
            #     kernel_size=(1, 1, 1),
            #     bias=True,
            # )
            # self.contrastive_head = torch.nn.Conv3d(
            #     in_channels=2048, out_channels=512, kernel_size=(1, 1, 1), bias=True
            # )
        else:
            final_time_dim = int(math.ceil(frame_nb / 16))
            self.avgpool = inflate.inflate_pool(
                resnet2d.avgpool, time_dim=final_time_dim
            )
            self.fc = inflate.inflate_linear(resnet2d.fc, 1)

    def forward(self, x):
        skips = []
        # ipdb.set_trace()
        x = x.permute(0, 1, 4, 2, 3)
        x = torch.cat((x, x, x), dim=1)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.maxpool(x)

        # 原版
        # x = checkpoint.checkpoint(self.layer1, x)
        # # ipdb.set_trace()
        # if self.return_skips:
        #     skips.append(x.permute(0, 1, 3, 4, 2))
        # x = checkpoint.checkpoint(self.layer2, x)
        # # ipdb.set_trace()
        # if self.return_skips:
        #     skips.append(x.permute(0, 1, 3, 4, 2))
        # x = checkpoint.checkpoint(self.layer3, x)
        # # ipdb.set_trace()
        # if self.return_skips:
        #     skips.append(x.permute(0, 1, 3, 4, 2))
        # x = checkpoint.checkpoint(self.layer4, x)
        # # ipdb.set_trace()
        # if self.return_skips:
        #     skips.append(x.permute(0, 1, 3, 4, 2))

        # 删掉了checkpoint
        # x = checkpoint.checkpoint(self.layer1, x)
        x = self.layer1(x)
        # ipdb.set_trace()
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        # x = checkpoint.checkpoint(self.layer2, x)
        x = self.layer2(x)
        # ipdb.set_trace()
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        # x = checkpoint.checkpoint(self.layer3, x)
        x = self.layer3(x)
        # ipdb.set_trace()
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        # x = checkpoint.checkpoint(self.layer4, x)
        x = self.layer4(x)
        # ipdb.set_trace()
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))

        # ipdb.set_trace()
        # return x
                    
        if self.conv_class:
            x_features = self.avgpool(x)
            # ipdb.set_trace()
            
            if self.ImageEmbedding:
                return x_features.squeeze(2).squeeze(2).squeeze(2).unsqueeze(0)
            
            x_ehr = self.classifier(x_features)
            x_ehr = x_ehr.squeeze(3)
            x_ehr = x_ehr.squeeze(3)
            x_ehr = x_ehr.mean(2)
            x_contrastive = self.contrastive_head(x_features)
            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.mean(2)
            if self.return_skips:
                return x_contrastive, x_ehr, skips
            else:
                return x_contrastive, x_ehr
        else:
            x = self.avgpool(x)
            x_reshape = x.view(x.size(0), -1)
            x = self.fc(x_reshape)
        return x

def inflate_reslayer(reslayer2d):
    reslayers3d = []
    for layer2d in reslayer2d:
        layer3d = Bottleneck3d(layer2d)
        reslayers3d.append(layer3d)
    return torch.nn.Sequential(*reslayers3d)


class Bottleneck3d(torch.nn.Module):
    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        spatial_stride = bottleneck2d.conv2.stride[0]

        self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1, center=True)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = inflate.inflate_conv(
            bottleneck2d.conv2,
            time_dim=3,
            time_padding=1,
            time_stride=spatial_stride,
            center=True,
        )
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)

        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1, center=True)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)

        self.relu = torch.nn.ReLU(inplace=False)

        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(
                bottleneck2d.downsample, time_stride=spatial_stride
            )
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        def run_function(input_x):
            out = self.conv1(input_x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            return out
        
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        # if x.requires_grad:
        #     out = checkpoint.checkpoint(run_function, x)
        # else:
        #     out = run_function(x)
        out = run_function(x) # xiugai

        out = out + residual
        out = self.relu(out)
        return out

def inflate_downsample(downsample2d, time_stride=1):
    downsample3d = torch.nn.Sequential(
        inflate.inflate_conv(
            downsample2d[0], time_dim=1, time_stride=time_stride, center=True
        ),
        inflate.inflate_batch_norm(downsample2d[1]),
    )
    return downsample3d
