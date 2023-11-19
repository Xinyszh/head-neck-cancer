import torch
import torch.nn as nn
from module.resnet3d import resnet18, resnet34, resnet50,resnet10
from module.feat_fuse import FeatFuse, TransFuse, IntraInterTransFuse
# from module.vit3d import ViT3D
# from module.transformer import TextTransformer
# from module.swin_transformer import SwinTrans3D
from monai.utils import ensure_tuple_rep
# from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForSequenceClassification


def get_model(model_tag, num_classes, data_size):
    if model_tag == 'ResNet10':
        model = resnet10(input_channel=data_size[0], num_classes=num_classes)
    elif model_tag == "ResNet18":
        model = resnet18(input_channel=data_size[0], num_classes=num_classes)
    elif model_tag == "ResNet34":
        model = resnet34(input_channel=data_size[0], num_classes=num_classes)
    elif model_tag == "ResNet50":
        model = resnet50(input_channel=data_size[0], num_classes=num_classes)
    elif model_tag == 'ViT':
        model = ViT3D(data_size=data_size,  # image size (6, 384, 256, 128)
                      num_classes=num_classes,
                      data_patch_size=16,  # data patch size
                      dim=384,
                      num_layer=12,
                      heads=6,
                      mlp_dim=768,
                      dropout=0.1,
                      emb_dropout=0.1)
    elif model_tag == 'SwinViT':
        model = SwinTrans3D(in_channels=data_size[0],
                            feature_size=48,
                            window_size=ensure_tuple_rep(7, 3),
                            patch_size=ensure_tuple_rep(2, 3),
                            dropout_path_rate=0.0,
                            use_checkpoint=True,
                            spatial_dims=3,
                            num_classes=num_classes,
                            dim=768)

        state_dict = torch.load('/home/luyang/LM/research-contributions/SwinUNETR/Pretrain/model_swinvit.pt')
        model.load_from(state_dict)
        # for param in model.parameters():
        #    param.requires_grad = False
        # for param in model.fc.parameters():
        #    param.requires_grad = True
    elif model_tag == 'TextTransformer':
        # model = TextTransformer(embed_dim=768,
        #                    depth=4, 
        #                    num_heads=4, 
        #                    mlp_ratio=2,
        #                    num_patches=500,
        #                    num_classes=num_classes,
        #                    drop_rate=0.)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=num_classes)
        model.train()
        # for param in model.bert.parameters():
        #    param.requires_grad = False
    else:
        raise NotImplementedError
    return model


def get_multimodal_model(fuse_tag, models_tag, data_sizes, num_classes):
    before_model_tag, after_model_tag = models_tag
    before_data_size, after_data_size = data_sizes

    before_model = get_model(before_model_tag, num_classes, before_data_size)
    after_model = get_model(after_model_tag, num_classes, after_data_size)


    if fuse_tag == 'Concate':
        model = FeatFuse(before_model, after_model, num_classes=num_classes)
    elif fuse_tag == 'Transformer':
        model = TransFuse(before_model, after_model,  num_classes=num_classes)
    elif fuse_tag == 'IntraInterTransFuse':
        model = IntraInterTransFuse(before_model, after_model, num_classes=num_classes)
    return model


def get_optimizer(optimizer_tag, model, lr, weight_decay):
    if optimizer_tag == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    elif optimizer_tag == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    elif optimizer_tag == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9
        )
    return optimizer