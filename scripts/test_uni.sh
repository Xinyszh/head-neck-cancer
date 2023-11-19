python test_unimodal.py --modal DCE --bs 1 --model_tag ResNet34\
                --seed 1 --device 0 --split additional_test --exp_name DCE-ce-cosine_lr-longer --save_csv

#python test_unimodal.py --modal T2 --bs 24 --model_tag ResNet18 --lr 1e-4  --split test\
#                --seed 1 --device 0 --exp_name T2

#python test_unimodal.py --modal DWI --bs 96 --model_tag ResNet18 --lr 1e-4  --split test\
#                --seed 1 --device 1 --exp_name DWI
