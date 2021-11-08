# test for ablation study
python test.py -net_arch RNN_mask -num_resblock 1 -num_cycle 5 -opt options/test/test_SRDenseNET.yml -in_channels 4 -trained_model experiments/MTF_ablation_study/RNN_MASK_in4f64_x4/epochs/best_ckp.pth
python test.py -net_arch RNN_Residual_mask -num_resblock 1 -num_cycle 5 -opt options/test/test_SRDenseNET.yml -in_channels 4 -trained_model experiments/MTF_ablation_study/RNN_RESIDUAL_MASK_in4f64_x4/epochs/best_ckp.pth
python test.py -net_arch RNN_Residual_Biv2_mask -num_resblock 1 -num_cycle 5 -opt options/test/test_SRDenseNET.yml -in_channels 4 -trained_model experiments/MTF_ablation_study/RNN_RESIDUAL_BIV2_MASK_in4f64_x4/epochs/best_ckp.pth
python test.py -net_arch RNN_Residual_Bi_pan_mask -num_resblock 1 -num_cycle 5 -opt options/test/test_SRDenseNET.yml -in_channels 4 -trained_model experiments/MTF_ablation_study/RNN_RESIDUAL_BI_PAN_MASK_in4f64_x4/epochs/best_ckp.pth
python test.py -net_arch RNN_Residual_Bi_pan_indepentv2_mask -num_resblock 1 -num_cycle 5 -opt options/test/test_SRDenseNET.yml -in_channels 4 -trained_model experiments/MTF_ablation_study/RNN_RESIDUAL_BI_PAN_INDEPENTV2_MASK_in4f64_x4/epochs/best_ckp.pth

# test

python test.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -opt options/test/test.yml -in_channels 4 -trained_model experiments/RNN_RESIDUAL_BI_PAN_FB_MASK_in4f64_x4_FIX4Rand/epochs/best_ckp.pth
python test.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -opt options/test/test.yml -in_channels 4 -trained_model experiments/ArbRPN_mix4sort_batch16_lr0002/epochs/best_ckp.pth

CUDA_VISIBLE_DEVICES=0, python test.py -net_arch RNN_Residual_Bi_pan_fb_mask -datamode LRHR -num_resblock 1 -num_cycle 5 -opt options/test/test.yml -trained_model experiments/train_strategies/rangFIX1-4/epochs/best_ckp.pth
CUDA_VISIBLE_DEVICES=0, python test.py -net_arch RNN_Residual_Bi_pan_fb_mask -datamode LRHR -num_resblock 1 -num_cycle 5 -opt options/test/test.yml -trained_model experiments/ArbRPN_repeat8_batch16_lr00002/epochs/best_ckp.pth

CUDA_VISIBLE_DEVICES=0, python test.py -net_arch RNN_Residual_Bi_pan_fb_mask -datamode LRHR -num_resblock 3 -num_cycle 5 -opt options/test/test.yml -trained_model experiments/big_ArbRPN_WV2_FIX8/epochs/best_ckp.pth
CUDA_VISIBLE_DEVICES=0, python test.py -net_arch BIEDN -datamode LRHRLRPAN -opt options/test/test.yml -trained_model experiments/SOTA_WV2/BIEDN_in8f64_x4/epochs/best_ckp.pth

# alternative-training-iteration
CUDA_VISIBLE_DEVICES=0, python test.py -net_arch RNN_Residual_Bi_pan_fb_mask -datamode LRHR -num_resblock 1 -num_cycle 5 -opt options/test/test.yml -trained_model experiments/train_strategies/rangFIX1-4-iteration/epochs/best_ckp.pth

# vanilla BiRNN +RL
CUDA_VISIBLE_DEVICES=0, python test.py -net_arch paraBiRNN -datamode LRHR -num_resblock 3 -num_cycle 1 -opt options/test/test.yml -trained_model experiments/MTF_MIX_ablation_study/paraBiRNN/epochs/best_ckp.pth

# test ArbRPN-SD
python test.py -net_arch RNN_Residual_Bi_pan_indepentv2_mask -num_resblock 1 -num_cycle 5 -opt options/test/test.yml -in_channels 4 -trained_model experiments/MTF_ablation_study/RNN_RESIDUAL_BI_PAN_INDEPENTV2_MASK_in4f64_x4/epochs/best_ckp.pth

python test.py -net_arch RNN_Residual_Bi_pan_mask -num_resblock 1 -num_cycle 5 -opt options/test/test.yml -in_channels 4 -trained_model experiments/MTF_ablation_study/RNN_RESIDUAL_BI_PAN_MASK_in4f64_x4/epochs/best_ckp.pth


python test.py -net_arch ArbRPN -opt options/test/test.yml -trained_model models/ArbRPN_QB_MIX4.pth