# ------------------ train multimodal model with t2, dwi, and dce ------------------ #
python train_multimodal.py --modal Multi --fuse_tag Concate --T2_model_tag ResNet18 --DWI_model_tag ResNet18 --DCE_model_tag ResNet34 --optimizer_tag Adam\
        --bs 7 --lr 3e-4 --lowest_lr 1e-6 --num_epochs 240 --seed 1 --device 3 --exp_name Multimodal-featfuse-2_redo

