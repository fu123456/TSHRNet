#!/bin/bash

## run with "bash -i dependencies_install.sh"
## if it does not work with errors, please run line by line in shell

conda create --yes --name TSHRNet python=3.9
conda activate TSHRNet
conda install --yes pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install --yes tqdm matplotlib
