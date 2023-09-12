from utils.data_loader_four_tuples_mix import ImageDataset, ImageTransform, generate_training_data_list, generate_testing_data_list

from models.UNet import UNet
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import models
from torchvision import transforms
from torch.autograd import Variable
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import time
import torch
import os

torch.manual_seed(1)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', type=str, default=None, help='the number of checkpoints')
    parser.add_argument('-s', '--image_size', type=int, default=256)
    parser.add_argument('-tedd', '--testing_data_dir', type=str, default='dataset/shapenet_specular_1500/testing_data')
    parser.add_argument('-tedlf', '--testing_data_list_file', type=str, default='dataset/shapenet_specular_1500/test.lst')
    parser.add_argument('-mn', '--model_name', type=str, default='SSHR')
    parser.add_argument('-tdn', '--testing_data_name', type=str, default='SSHR')
    return parser


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict


def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor((0.5, )) + torch.Tensor((0.5, ))
    x = x.transpose(1, 3)
    return x


def test(UNet1, UNet2, UNet3, UNet4, model_name, test_dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    UNet1.to(device)
    UNet2.to(device)
    UNet3.to(device)
    UNet4.to(device)

    if device == 'cuda':
        UNet1 = torch.nn.DataParallel(UNet1)
        UNet2 = torch.nn.DataParallel(UNet2)
        UNet3 = torch.nn.DataParallel(UNet3)
        UNet4 = torch.nn.DataParallel(UNet4)
        print("parallel mode")

    print("device:{}".format(device))

    UNet1.eval()
    UNet2.eval()
    UNet3.eval()
    UNet4.eval()

    # dirs for saving results
    dir_t1 = './test_result_' + model_name + '/' + key_dir + '/' + 'estimated_albedo'
    dir_t2 = './test_result_' + model_name + '/' + key_dir + '/' + 'estimated_shading'
    dir_t3 = './test_result_' + model_name + '/' + key_dir + '/' + 'grid'
    dir_t4 = './test_result_' + model_name + '/' + key_dir + '/' + 'estimated_diffuse'
    dir_t5 = './test_result_' + model_name + '/' + key_dir + '/' + 'estimated_diffuse_tc'

    for n, (input_img, gt_specular_residue, gt_diffuse, gt_diffuse_tc) in enumerate([test_dataset[i] for i in range(test_dataset.__len__())]):

        # print(test_dataset.img_list['path_i'][n].split('/')[4][:-4])

        input_img = torch.unsqueeze(input_img, dim=0)
        gt_diffuse = torch.unsqueeze(gt_diffuse, dim=0)
        gt_diffuse_tc = torch.unsqueeze(gt_diffuse_tc, dim=0)

        with torch.no_grad():
            estimated_diffuse = UNet1(input_img.to(device))
            estimated_specular_residue = UNet2(input_img.to(device))
            # estimat diffuse
            G3_data = torch.cat([estimated_diffuse, input_img.to(device)], dim=1)
            estimated_diffuse_refined = UNet3(G3_data.to(device))

            # the third stage (tone correction)
            input_img = input_img.to(device)
            G4_input = torch.cat([estimated_diffuse_refined, estimated_specular_residue, input_img], dim=1)
            estimated_diffuse_tc = UNet4(G4_input.to(device))

            # to cpu
            estimated_diffuse = estimated_diffuse.to(torch.device('cpu'))
            estimated_diffuse_refined = estimated_diffuse_refined.to(torch.device('cpu'))
            estimated_diffuse_tc = estimated_diffuse_tc.to(torch.device('cpu'))
            input_img = input_img.to(torch.device('cpu'))


        grid= make_grid(torch.cat((unnormalize(input_img), unnormalize(gt_diffuse), unnormalize(estimated_diffuse_refined), unnormalize(estimated_diffuse_tc)), dim=0))

        temp = len(test_dataset.img_list['path_i'][n].split('/'))
        r_subdir = test_dataset.img_list['path_i'][n].split('/')[temp-2]
        basename = test_dataset.img_list['path_i'][n].split('/')[temp-1]
        print(r_subdir)
        print(basename)
        subdir = os.path.join(dir_t3, r_subdir)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        grid_name = os.path.join(subdir, basename)
        save_image(grid, grid_name)

        # save diffuse images
        estimated_diffuse_refined = transforms.ToPILImage(mode='RGB')(unnormalize(estimated_diffuse_refined)[0, :, :, :])
        subdir = os.path.join(dir_t4, r_subdir)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        estimated_diffuse_name = os.path.join(subdir, basename)
        estimated_diffuse_refined.save(estimated_diffuse_name)

        # save tone-corrected diffuse images
        estimated_diffuse_tc = transforms.ToPILImage(mode='RGB')(unnormalize(estimated_diffuse_tc)[0, :, :, :])
        subdir = os.path.join(dir_t5, r_subdir)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        estimated_diffuse_tc_name = os.path.join(subdir, basename)
        estimated_diffuse_tc.save(estimated_diffuse_tc_name)


def main(parser):
    UNet1 = UNet(input_channels=3, output_channels=3)
    UNet2 = UNet(input_channels=3, output_channels=3)
    UNet3 = UNet(input_channels=6, output_channels=3)
    UNet4 = UNet(input_channels=9, output_channels=3)

    if parser.load is not None:
        print('load checkpoint ' + parser.load)
        # load UNet weights
        UNet1_weights = torch.load('./checkpoints_'+parser.model_name+'/UNet1_'+parser.load+'.pth')
        UNet1.load_state_dict(fix_model_state_dict(UNet1_weights))
        UNet2_weights = torch.load('./checkpoints_'+parser.model_name+'/UNet2_'+parser.load+'.pth')
        UNet2.load_state_dict(fix_model_state_dict(UNet2_weights))
        UNet3_weights = torch.load('./checkpoints_'+parser.model_name+'/UNet3_'+parser.load+'.pth')
        UNet3.load_state_dict(fix_model_state_dict(UNet3_weights))
        UNet4_weights = torch.load('./checkpoints_'+parser.model_name+'/UNet4_'+parser.load+'.pth')
        UNet4.load_state_dict(fix_model_state_dict(UNet4_weights))

    mean = (0.5,)
    std = (0.5,)

    size = parser.image_size
    testing_data_dir = parser.testing_data_dir
    testing_data_list_file = parser.testing_data_list_file
    model_name = parser.model_name

    test_img_list = generate_testing_data_list(testing_data_dir, testing_data_list_file)
    test_dataset = ImageDataset(img_list=test_img_list, img_transform=ImageTransform(size=size, mean=mean, std=std), phase='test')
    test(UNet1, UNet2, UNet3, UNet4, model_name, test_dataset)

if __name__ == "__main__":
    parser = get_parser().parse_args()
    if parser.load is not None:
        load_num = str(parser.load)
        model_name = parser.model_name
        testing_data_name = parser.testing_data_name
        key_dir = testing_data_name + '_' + model_name + '_' + load_num # like SSHR_SSHR_60

    main(parser)
