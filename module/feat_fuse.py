import torch
import torch.nn as nn
from module.resnet3d import resnet18, resnet34, resnet50
# from module.vit3d import Transformer
from einops import rearrange, repeat


class IntraInterTransFuse(nn.Module):
    def __init__(self, before_model, after_model, num_classes=2):
        super(IntraInterTransFuse, self).__init__()

        self.features = FeatureExtractor(before_model, after_model)
        
        self.before_linear = nn.Conv3d(512, 256, kernel_size=1, stride=1, bias=False)
        self.after_linear = nn.Conv3d(512, 256, kernel_size=1, stride=1, bias=False)
    
        
        dim = 256
        self.before_transformer = Transformer(dim, num_layer=1, heads=8, dim_head=32, mlp_dim=256, dropout=0)
        self.before_pos_embedding = nn.Parameter(torch.randn(1, 192+1, dim))
        num_patches = 128 #608，384+96+128
        
        self.after_transformer = Transformer(dim, num_layer=1, heads=8, dim_head=32, mlp_dim=256, dropout=0)
        self.after_pos_embedding = nn.Parameter(torch.randn(1, 32, dim))
        
        
        self.Multi_transformer = Transformer(dim, num_layer=1, heads=8, dim_head=32, mlp_dim=512, dropout=0)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
            )

    def forward(self, x_1, x_2):
        x_1, x_2 = self.features(x_1, x_2)
        x_1 = self.before_linear(x_1)
        x_1 = x_1.flatten(2)
        x_1 = x_1.transpose(-1, -2)
        x_1 += self.before_pos_embedding[:, 1:]
        x_1 = self.before_transformer(x_1)
        x_2 = self.after_linear(x_2)
        x_2 = x_2.flatten(2)
        x_2 = x_2.transpose(-1, -2)
        x_2 += self.after_pos_embedding
        x_2 = self.after_transformer(x_2)
       
        
        x = torch.cat((x_1, x_2), dim=1)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embedding = torch.cat((self.before_pos_embedding, self.after_pos_embedding), dim=1)
        x += pos_embedding[:, :(n + 1)]
        
        x = self.Multi_transformer(x)
        x = x.mean(dim = 1)
        x = self.to_latent(x)
        
        return self.mlp_head(x)


class TransFuse(nn.Module):
    def __init__(self, before_model, after_model, num_classes=2):
        super(TransFuse, self).__init__()

        self.features = FeatureExtractor(before_model, after_model)
        
        self.before_linear = nn.Conv3d(512, 256, kernel_size=1, stride=1, bias=False)
        self.after_linear = nn.Conv3d(512, 256, kernel_size=1, stride=1, bias=False)
        
        dim = 256
        self.transformer = Transformer(dim, num_layer=2, heads=6, dim_head=64, mlp_dim=512, dropout=0)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        num_patches = 608
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
            )

    def forward(self, x_1, x_2):
        x_1, x_2= self.features(x_1, x_2)
        x_1 = self.before_linear(x_1)
        x_1 = x_1.flatten(2)
        x_1 = x_1.transpose(-1, -2)
        x_2 = self.before_linear(x_2)
        x_2 = x_2.flatten(2)
        x_2 = x_2.transpose(-1, -2)
    
        x = torch.cat((x_1, x_2), dim=1)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        
        x = self.transformer(x)
        x = x.mean(dim = 1)
        x = self.to_latent(x)
        
        return self.mlp_head(x)


class FeatFuse(nn.Module):
    def __init__(self, before_model, after_model, num_classes=2):
        super(FeatFuse, self).__init__()

        self.features = FeatureExtractor(before_model, after_model)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.classifier = nn.Linear(512*3, num_classes)
        self.classifier = nn.Sequential(nn.Linear(512*2, 512*2),
                                        nn.ReLU(),
                                        nn.Linear(512*2, 512*2),
                                        nn.ReLU(),
                                        nn.Linear(512*2, num_classes))

    def forward(self, x_1, x_2):
        x_1, x_2 = self.features(x_1, x_2)
        
        x_1 = torch.flatten(self.avgpool(x_1), 1)
        x_2 = torch.flatten(self.avgpool(x_2), 1)
       
        
        x = torch.cat((x_1, x_2), dim=1)
        x = self.classifier(x)

        return x
    
class FeatureExtractor(nn.Module):
    def __init__(self, before_model, after_model):
        super(FeatureExtractor, self).__init__()

        self.before_features = nn.Sequential(before_model.conv1,
                                   before_model.bn1,
                                   before_model.relu,
                                   before_model.maxpool,
                                   before_model.layer1,
                                   before_model.layer2,
                                   before_model.layer3,
                                   before_model.layer4,
                                   )
        self.after_features = nn.Sequential(after_model.conv1,
                                   after_model.bn1,
                                   after_model.relu,
                                   after_model.maxpool,
                                   after_model.layer1,
                                   after_model.layer2,
                                   after_model.layer3,
                                   after_model.layer4,
                                   )

        
    def forward(self, x_1, x_2):
        x_1 = self.before_features(x_1)
        x_2 = self.after_features(x_2)
        return x_1, x_2