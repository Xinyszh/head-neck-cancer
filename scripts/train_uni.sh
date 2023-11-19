python train_text.py --modal Text --bs 16 --model_tag TextTransformer --optimizer_tag AdamW\
               --lr 1e-3 --lowest_lr 1e-5 --num_epochs 120 --seed 1 --device 6 --exp_name Text

#python train_unimodal.py --modal DCE --bs 7 --model_tag ResNet34 --optimizer_tag Adam\
#               --num_epochs 240 --seed 1 --device 0 --exp_name DCE-ce-cosine_lr-longer

#python train_unimodal.py --modal T2 --bs 24 --model_tag ResNet18 --optimizer_tag Adam\
#               --num_epochs 100 --seed 1 --device 0 --exp_name T2

#python train_unimodal.py --modal DWI --bs 96 --model_tag ResNet18 --optimizer_tag Adam\
#               --num_epochs 100 --seed 1 --device 1 --exp_name DWI

# ViT
#python train_unimodal.py --modal DCE --bs 3 --model_tag ViT --optimizer_tag AdamW --lr 5e-2 --wd 1e-2\
#               --num_epochs 400 --seed 1 --device 0 --exp_name DCE-ViT-bigger

# Swin
#python train_unimodal.py --modal DWI --bs 10 --model_tag SwinViT --optimizer_tag AdamW --lr 3e-4 \
#               --num_epochs 100 --seed 1 --device 0 --exp_name DWI-Swin-try2