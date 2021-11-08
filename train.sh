#!/bin/bash

#python train.py -net_arch RNN_mask -num_layers 12 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_mask -num_layers 12 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Biv2_mask -num_layers 12 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pan_mask -num_layers 12 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pan_indepentv2_mask -num_layers 12 -opt options/train/train_demo.yml


#python train.py -net_arch RNN_Residual_Bi_pan_mask -num_layers 4 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pan_mask -num_layers 8 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pan_mask -num_layers 12 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pan_mask -num_layers 16 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pan_mask -num_layers 20 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pan_mask -num_layers 24 -opt options/train/train_demo.yml

#python train.py -net_arch RNN_Residual_Bi_hh_mask -num_layers 12 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_hp_mask -num_layers 12 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_ph_mask -num_layers 12 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pp_mask -num_layers 12 -opt options/train/train_demo.yml


#python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 1 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 2 -num_cycle 1 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 3 -num_cycle 1 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 4 -num_cycle 1 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 5 -num_cycle 1 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 6 -num_cycle 1 -opt options/train/train_demo.yml

#python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 1 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 2 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 3 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 4 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -opt options/train/train_demo.yml
#python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 6 -opt options/train/train_demo.yml



#python train.py -net_arch PNN -opt options/train/train_demo.yml
#python train.py -net_arch DRPNN -opt options/train/train_demo.yml
#python train.py -net_arch PanNet -opt options/train/train_demo.yml
#python train.py -net_arch BDPN -opt options/train/train_demo.yml
#python train.py -net_arch TFNET   -opt options/train/train_demo.yml
#python train.py -net_arch DILATED -opt options/train/train_demo.yml
#python train.py -net_arch DiCNN1 -opt options/train/train_demo.yml
#python train.py -net_arch MSDCNN -opt options/train/train_demo.yml
#python train.py -net_arch MSDCNN -opt options/train/train_demo.yml -in_channels 4 -lrpath /media/clh/Share/PanSharpening_Methods/PanSharp_dataset/QB_20200922/train/MTF/4bands_sort/LRMS_npy
#python train.py -net_arch MSDCNN -opt options/train/train_demo.yml -in_channels 4 -lrpath /media/clh/Files/PanSharp_dataset_20200922/IK/train/MTF/4bands/LRMS_npy
#python train.py -net_arch BDPN -loss myloss -opt options/train/train_demo.yml -in_channels 8 -datamode LRHRLRPAN -lrpath ../PanSharp_dataset/WV2/train/MTF/8bands/LRMS_npy
#python train.py -net_arch BDPN -loss myloss -opt options/train/train_demo.yml -in_channels 4 -datamode LRHRLRPAN -lrpath /media/clh/Share/PanSharpening_Methods/PanSharp_dataset/QB_20200922/train/MTF/4bands_sort/LRMS_npy
#python train.py -net_arch BDPN -loss myloss -opt options/train/train_demo.yml -in_channels 4 -datamode LRHRLRPAN -lrpath /media/clh/Files/PanSharp_dataset_20200922/IK/train/MTF/4bands/LRMS_npy

#python train.py -net_arch BDPN -loss l1 -opt options/train/train_demo.yml -in_channels 8 -lrpath ../PanSharp_dataset/WV2/train/MTF/8bands/LRMS_npy
#python train.py -net_arch BDPN -loss l1 -opt options/train/train_demo.yml -in_channels 4 -lrpath /media/clh/Share/PanSharpening_Methods/PanSharp_dataset/QB_20200922/train/MTF/4bands_sort/LRMS_npy
#python train.py -net_arch BDPN -loss l1 -opt options/train/train_demo.yml -in_channels 4 -lrpath /media/clh/Files/PanSharp_dataset_20200922/IK/train/MTF/4bands/LRMS_npy

#python train.py -net_arch BIEDN -opt options/train/train_demo.yml -in_channels 8 -datamode LRHRLRPAN -lrpath ../PanSharp_dataset/WV2/train/MTF/8bands/LRMS_npy
#python train.py -net_arch BIEDN -opt options/train/train_demo.yml -in_channels 4 -datamode LRHRLRPAN -lrpath /media/clh/Share/PanSharpening_Methods/PanSharp_dataset/QB_20200922/train/MTF/4bands_sort/LRMS_npy
#python train.py -net_arch BIEDN -opt options/train/train_demo.yml -in_channels 4 -datamode LRHRLRPAN -lrpath /media/clh/Files/PanSharp_dataset_20200922/IK/train/MTF/4bands/LRMS_npy

#python train.py -net_arch PanNet -opt options/train/train_demo.yml -in_channels 8 -lrpath ../PanSharp_dataset/WV2/train/MTF/8bands/LRMS_npy
#python train.py -net_arch PanNet -opt options/train/train_demo.yml -in_channels 4 -lrpath /media/clh/Share/PanSharpening_Methods/PanSharp_dataset/QB_20200922/train/MTF/4bands_sort/LRMS_npy
#python train.py -net_arch PanNet -opt options/train/train_demo.yml -in_channels 4 -lrpath /media/clh/Files/PanSharp_dataset_20200922/IK/train/MTF/4bands/LRMS_npy

#python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -loss myloss -opt options/train/train_demo.yml -in_channels 8 -lrpath ../PanSharp_dataset/WV2/train/MTF/8bands/LRMS_npy
#python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -loss myloss -opt options/train/train_demo.yml -in_channels 4 -lrpath ../PanSharp_dataset/QB_20200922/train/MTF/4bands_sort/LRMS_npy
#python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -loss myloss -opt options/train/train_demo.yml -in_channels 4 -lrpath /media/clh/Files/PanSharp_dataset_20200922/IK/train/MTF/4bands/LRMS_npy


# python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -loss myloss -opt options/train/train_demo.yml -in_channels 4 -lrpath ../PanSharp_dataset/QB_20200922/train/MTF/1bands_rand/LRMS_npy -panpath ../PanSharp_dataset/QB_20200922/train/MTF/4bands_sort/LRPAN_npy
# python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -loss myloss -opt options/train/train_demo.yml -in_channels 4 -lrpath ../PanSharp_dataset/QB_20200922/train/MTF/2bands_rand/LRMS_npy -panpath ../PanSharp_dataset/QB_20200922/train/MTF/4bands_sort/LRPAN_npy
# python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -loss myloss -opt options/train/train_demo.yml -in_channels 4 -lrpath ../PanSharp_dataset/QB_20200922/train/MTF/3bands_rand/LRMS_npy -panpath ../PanSharp_dadtaset/QB_20200922/train/MTF/4bands_sort/LRPAN_npy
#python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -loss myloss -opt options/train/train_demo.yml -in_channels 4 -lrpath ../PanSharp_dataset/QB_20200922/train/MTF/4bands_rand/LRMS_npy -panpath ../PanSharp_dataset/QB_20200922/train/MTF/4bands_sort/LRPAN_npy
# python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -loss myloss -opt options/train/train_demo.yml -in_channels 4 -lrpath ../PanSharp_dataset/QB_20200922/train/MTF/4bands_sort/LRMS_npy -panpath ../PanSharp_dataset/QB_20200922/train/MTF/4bands_sort/LRPAN_npy
# python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -loss myloss -opt options/train/train_demo.yml -in_channels 4 -lrpath ../PanSharp_dataset/QB_20200922/train/MTF/3bands_sort/LRMS_npy -panpath ../PanSharp_dataset/QB_20200922/train/MTF/4bands_sort/LRPAN_npy
# python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -loss myloss -opt options/train/train_demo.yml -in_channels 4 -lrpath ../PanSharp_dataset/QB_20200922/train/MTF/2bands_sort/LRMS_npy -panpath ../PanSharp_dataset/QB_20200922/train/MTF/4bands_sort/LRPAN_npy


############ ablation study on MIX-dataset ############
# CUDA_VISIBLE_DEVICES=0, nohup python train.py -net_arch RNN_mask -num_resblock 1 -num_cycle 5 -opt options/train/train_demo.yml -in_channels 4 > RNN_MIX4.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0, nohup python train.py -net_arch RNN_Residual_mask -num_resblock 1 -num_cycle 5 -opt options/train/train_demo.yml -in_channels 4 > RNN_RL_MIX4.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0, nohup python train.py -net_arch RNN_Residual_Biv2_mask -num_resblock 1 -num_cycle 5 -opt options/train/train_demo.yml -in_channels 4 > BiRNN_LR_MIX4.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0, nohup python train.py -net_arch RNN_Residual_Bi_pan_mask -num_resblock 1 -num_cycle 5 -opt options/train/train_demo.yml -in_channels 4 > ArbRPN_MIX4.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0, nohup python train.py -net_arch RNN_Residual_Bi_pan_indepentv2_mask -num_resblock 1 -num_cycle 5 -opt options/train/train_demo.yml -in_channels 4  -lrpath ../PanSharp_dataset/QB/train/MTF/MIX_4_rand/LRMS_npy -panpath ../PanSharp_dataset/QB/train/MTF/4bands/LRPAN_npy > ArbRPN_SD_MIX4.txt 2>&1 &

########### train with different training strategies #############
# python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -loss myloss -opt options/train/train_demo.yml -in_channels 4 > mix4sort.txt 2>&1 &
# python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -loss myloss -opt options/train/train_demo.yml -in_channels 4 -save_name FIX4Rand -lrpath ../PanSharp_dataset/QB/train/MTF/4bands_rand/LRMS_npy -panpath ../PanSharp_dataset/QB/train/MTF/4bands/LRPAN_npy > FIX4Rand.txt 2>&1 &
# python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -loss myloss -opt options/train/train_demo.yml -in_channels 4 -save_name mix4sort_batch16_lr0002 > mix4sort_batch16_lr0002.txt 2>&1 &

# nohup python train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -loss myloss -opt options/train/train_demo.yml -in_channels 4 -save_name train_strategies/rangFIX1-4 > rangFIX1-4.txt 2>&1 &

# nohup python train.py -net_arch paraBiRNN -num_resblock 3 -num_cycle 1 -loss myloss -opt options/train/train_demo.yml -in_channels 4 -save_name paraBiRNN > paraBiRNN.txt 2>&1 &

# nohup python train_iterations.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -loss myloss -opt options/train/train_demo.yml -in_channels 4 -save_name train_strategies/rangFIX1-4-iteration > rangFIX1-4-iteration.txt 2>&1 &

# nohup python train.py -net_arch paraBiRNN -num_resblock 3 -num_cycle 1 -loss myloss -opt options/train/train_demo.yml -in_channels 4 -save_name paraBiRNN > paraBiRNN.txt 2>&1 &
# train.py -net_arch RNN_Residual_Bi_pan_fb_mask -num_resblock 1 -num_cycle 5 -loss myloss -opt options/train/train_demo.yml -in_channels 4 > mix4sort.txt 2>&1 &

# sleep 210m
python train.py -net_arch TFNET  -loss l1 -opt options/train/train_demo.yml -in_channels 4 -save_name SOTA_NBU_WV4/TFNet
python train.py -net_arch PNN  -loss l1 -opt options/train/train_demo.yml -in_channels 4 -save_name SOTA_NBU_WV4/PNN
python train.py -net_arch MSDCNN  -loss l1 -opt options/train/train_demo.yml -in_channels 4 -save_name SOTA_NBU_WV4/MSDCNN
python train.py -net_arch DRPNN  -loss l1 -opt options/train/train_demo.yml -in_channels 4 -save_name SOTA_NBU_WV4/DRPNN
python train.py -net_arch DiCNN1  -loss l1 -opt options/train/train_demo.yml -in_channels 4 -save_name SOTA_NBU_WV4/DICNN1
python train.py -net_arch PanNet  -loss l1 -opt options/train/train_demo.yml -in_channels 4 -save_name SOTA_NBU_WV4/PANNET
python train.py -net_arch DILATED  -loss l1 -opt options/train/train_demo.yml -in_channels 4 -save_name SOTA_NBU_WV4/DILATED
python train.py -net_arch ArbRPN -loss myloss -num_resblock 3 -num_cycle 5 -opt options/train/train_demo.yml -save_name SOTA_NBU_WV4/ArbRPN_woInit