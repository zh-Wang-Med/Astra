import copy

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import wordpunct_tokenize
import torchvision
import ipdb
from merlin.models import i3res

class Bottleneck3d_modified(nn.Module):
    expansion = 4  # ResNet Bottleneck的通道扩展系数

    def __init__(self, inplanes, planes, stride=1, downsample=None, time_stride=1):
        """
        直接定义3D Bottleneck模块。

        Args:
            inplanes (int): 输入通道数。
            planes (int): 基础通道数（输出通道数为 planes * self.expansion）。
            stride (int): 空间维度上的步长。
            downsample (nn.Module, optional): Downsample层。
            time_stride (int): 时间维度上的步长。
        """
        super(Bottleneck3d_modified, self).__init__()

        # Conv1: 1x1x1 conv
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        # Conv2: 3x3x3 conv
        # 注意：时间步长(time_stride)和空间步长(stride)在这里应用
        self.conv2 = nn.Conv3d(
            planes, planes,
            kernel_size=(3, 3, 3),
            stride=(time_stride, stride, stride),
            padding=(1, 1, 1), # 时间和空间都填充1
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)

        # Conv3: 1x1x1 conv, 扩展通道
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride # 主要用于信息记录，实际步长在conv2和downsample中

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ImageEncoder(nn.Module):
    def __init__(self, ImageEmbedding: bool = False):
        super().__init__()
        self.ImageEmbedding = ImageEmbedding
        resnet = torchvision.models.resnet152(pretrained=False)
        self.i3_resnet = i3res.I3ResNet(
            copy.deepcopy(resnet), class_nb=1692, conv_class=True, ImageEmbedding=self.ImageEmbedding
        )
        del resnet

    def forward(self, image):
        if self.ImageEmbedding:
            contrastive_features = self.i3_resnet(image)
            return contrastive_features
        else:
            contrastive_features, ehr_features = self.i3_resnet(image)
            return contrastive_features, ehr_features
        
class ImageEncoder_modified(nn.Module):
    def __init__(self, ImageEmbedding: bool = False):
        super().__init__()
        self.ImageEmbedding = ImageEmbedding
        self.i3_resnet = i3res.I3ResNet_modified(block=Bottleneck3d_modified, layers=[3, 8, 36, 3], class_nb=1692)

    def forward(self, image):
        if self.ImageEmbedding:
            contrastive_features = self.i3_resnet(image)
            return contrastive_features
        else:
            contrastive_features, ehr_features = self.i3_resnet(image)
            return contrastive_features, ehr_features

class ImageEncoder_classify(nn.Module):
    def __init__(self, ImageEmbedding: bool = False):
        super().__init__()
        self.ImageEmbedding = ImageEmbedding
        resnet = torchvision.models.resnet152(pretrained=False)
        self.i3_resnet = i3res.I3ResNet_classify(
            copy.deepcopy(resnet), class_nb=1692, conv_class=True, ImageEmbedding=self.ImageEmbedding
        )
        del resnet

    def forward(self, image):
        if self.ImageEmbedding:
            contrastive_features = self.i3_resnet(image)
            return contrastive_features
        else:
            contrastive_features, ehr_features = self.i3_resnet(image)
            return contrastive_features, ehr_features


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        self.text_encoder = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
        self.text_encoder.gradient_checkpointing_enable()
        self.linear_layer = nn.Linear(768, 512)

    def forward(self, text_labels):
        text_labels = [sanitize_report(text) for text in text_labels]
        inputs = self.tokenizer(
            text_labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        text_embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
        text_embeddings = self.linear_layer(text_embeddings)
        return text_embeddings


class MerlinArchitecture(nn.Module):
    def __init__(self, init_logit_scale: float = 1.0, ImageEmbedding: bool = False): 
        super().__init__()
        self.ImageEmbedding = ImageEmbedding
        self.encode_image = ImageEncoder(ImageEmbedding=self.ImageEmbedding)
        # self.encode_text = TextEncoder()
        # self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    def forward(self, image, text=None):
        if self.ImageEmbedding and text is None:
            image_features = self.encode_image(image)
            return image_features
        elif self.ImageEmbedding and text is not None:
            raise ValueError("Text input not required for image embedding")
        elif text is None:
            raise ValueError("Text input required for Image and Text embedding")
        
        image_features, ehr_features = self.encode_image(image)
        text_features = self.encode_text(text)

        if len(image_features.shape) == 1:
            image_features = image_features.unsqueeze(0)
        if len(text_features.shape) == 1:
            text_features = text_features.unsqueeze(0)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return (
            image_features,
            ehr_features,
            text_features,
        )
    
class MerlinArchitecture_modified(nn.Module):
    def __init__(self, init_logit_scale: float = 1.0, ImageEmbedding: bool = False): 
        super().__init__()
        self.ImageEmbedding = ImageEmbedding
        self.encode_image = ImageEncoder_modified(ImageEmbedding=self.ImageEmbedding)
        # self.encode_text = TextEncoder()
        # self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    def forward(self, image, text=None):
        if self.ImageEmbedding and text is None:
            image_features = self.encode_image(image)
            return image_features
        elif self.ImageEmbedding and text is not None:
            raise ValueError("Text input not required for image embedding")
        elif text is None:
            raise ValueError("Text input required for Image and Text embedding")
        
        image_features, ehr_features = self.encode_image(image)
        text_features = self.encode_text(text)

        if len(image_features.shape) == 1:
            image_features = image_features.unsqueeze(0)
        if len(text_features.shape) == 1:
            text_features = text_features.unsqueeze(0)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return (
            image_features,
            ehr_features,
            text_features,
        )

class MerlinArchitecture_classify(nn.Module):
    def __init__(self, init_logit_scale: float = 1.0, ImageEmbedding: bool = False): 
        super().__init__()
        self.ImageEmbedding = ImageEmbedding
        self.encode_image = ImageEncoder_classify(ImageEmbedding=self.ImageEmbedding)
        # self.encode_text = TextEncoder()
        # self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    def forward(self, image, text=None):
        # ipdb.set_trace()
        if self.ImageEmbedding and text is None:
            image_features = self.encode_image(image)
            return image_features
        elif self.ImageEmbedding and text is not None:
            raise ValueError("Text input not required for image embedding")
        elif text is None:
            raise ValueError("Text input required for Image and Text embedding")
        
        image_features, ehr_features = self.encode_image(image)
        text_features = self.encode_text(text)

        if len(image_features.shape) == 1:
            image_features = image_features.unsqueeze(0)
        if len(text_features.shape) == 1:
            text_features = text_features.unsqueeze(0)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return (
            image_features,
            ehr_features,
            text_features,
        )




def sanitize_report(report):
    report = report.lower()
    return " ".join(wordpunct_tokenize(report))
