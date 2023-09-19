#!/bin/bash

set -e

# In default, we use the model trained on SSHR (or SHIQ or PSD) to process the testing images of SSHR (or SHIQ or PSD).
# The variable of "model_name" can be SSHR or SHIQ or PSD or mix_SSHR_SHIQ_PSD.


## >>> testing SSHR >>>
# due to out of memory, we split "test.lst" into four parts for testing
num_checkpoint=60 # the indexes of the used checkpoints
model_name='SSHR' # find the checkpoints in "checkpoints_${model_name}, like "checkpoints_SSHR"
testing_data_name='SSHR' # testing dataset name
python test_4_networks.py -mn ${model_name} -l ${num_checkpoint} -tdn ${testing_data_name} -tedd 'dataset' -tedlf 'dataset/SSHR/test_7_tuples_part1.lst'
python test_4_networks.py -mn ${model_name} -l ${num_checkpoint} -tdn ${testing_data_name} -tedd 'dataset' -tedlf 'dataset/SSHR/test_7_tuples_part2.lst'
python test_4_networks.py -mn ${model_name} -l ${num_checkpoint} -tdn ${testing_data_name} -tedd 'dataset' -tedlf 'dataset/SSHR/test_7_tuples_part3.lst'
python test_4_networks.py -mn ${model_name} -l ${num_checkpoint} -tdn ${testing_data_name} -tedd 'dataset' -tedlf 'dataset/SSHR/test_7_tuples_part4.lst'
## <<< testing SSHR <<<


# ## >>> testing SHIQ >>>
# num_checkpoint=60
# model_name='SHIQ'
# testing_data_name='SHIQ'
# python test_4_networks_mix.py -mn ${model_name} -l ${num_checkpoint} -tdn ${testing_data_name} -tedd 'dataset' -tedlf 'dataset/SHIQ_data_10825/test.lst'
# ## <<< testing SHIQ <<<


# ## >>> testing PSD >>>
# num_checkpoint=60
# model_name='PSD'
# testing_data_name='PSD'
# python test_4_networks_mix.py -mn ${model_name} -l ${num_checkpoint} -tdn ${testing_data_name} -tedd 'dataset' -tedlf 'dataset/PSD/test.lst'
# ## <<< testing PSD <<<
