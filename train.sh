#!/bin/bash

set -e

# to train on our SSHR dataset
python train_4_networks.py \
       -trdd dataset \
       -trdlf dataset/SSHR/train_7_tuples.lst \
       -dn SSHR

# # to train on SHIQ dataset
# python train_4_networks_mix.py \
    #        -trdd dataset \
    #        -trdlf dataset/SHIQ_data_10825/train.lst \
    #        -dn SHIQ

# # to train on PSD dataset
# python train_4_networks_mix.py \
    #        -trdd dataset\
    #        -trdlf dataset/PSD/train.lst \
    #        -dn PSD


# to train on the mix data of SSHR, SHIQ, and PSD, which could produce better results for real images
# generate list file

# if [ ! -f "dataset/train_mix.lst" ]; then
#     cat dataset/SSHR/train_4_tuples.lst dataset/SHIQ_data_10825/train.lst dataset/PSD/train.lst >> dataset/train_mix.lst
# else
#     rm -rf dataset/train_mix.lst
#     cat dataset/SSHR/train_4_tuples.lst dataset/SHIQ_data_10825/train.lst dataset/PSD/train.lst >> dataset/train_mix.lst
# fi
# python train_4_networks_mix.py \
    #        -trdd dataset \
    #        -trdlf dataset/train_mix.lst \
    #        -dn mix_SSHR_SHIQ_PSD
