#python train_unimodal.py --debug --modal DCE --bs 3 --model_tag ViT --optimizer_tag AdamW --lr 5e-3 --wd 1e-2\
#               --num_epochs 400 --seed 1 --device 0 --exp_name DCE-ViT-bigger

#python train_unimodal.py --debug --modal DWI --bs 1 --model_tag SwinViT --optimizer_tag AdamW --lr 5e-3 --wd 1e-2\
#               --num_epochs 400 --seed 1 --device 1 --exp_name DWI-ViT-bigger

#python train_multimodal.py --debug --modal Multi --fuse_tag Concate --text --T2_model_tag ResNet18 --DWI_model_tag ResNet18 --DCE_model_tag ResNet34 --optimizer_tag Adam\
#        --bs 5 --lr 1e-4 --num_epochs 120 --seed 1 --device 0 --exp_name debug

python train_text.py --debug --modal Text --bs 16 --model_tag TextTransformer --optimizer_tag AdamW\
               --lr 1e-5 --lowest_lr 1e-6 --num_epochs 240 --seed 1 --device 5 --exp_name debug