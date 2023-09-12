from utils.data_loader_seven_tuples import ImageDataset, ImageTransform, generate_training_data_list, generate_testing_data_list
from utils.fg_tools import fix_model_state_dict, plot_log, check_dir, unnormalize

from models.UNet import UNet
from tqdm import tqdm
from torchvision import transforms
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
    parser.add_argument('-ne', '--num_epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('-lne', '--load_num_epoch', type=str, default=None, help='the number of checkpoints')
    parser.add_argument('-s', '--image_size', type=int, default=256)
    parser.add_argument('-cs', '--crop_size', type=int, default=256)
    parser.add_argument('-lr', '--lr', type=float, default=1e-4)
    parser.add_argument('-pdn', '--pretrained_dataset_name', type=str, default=None, help='pretrained model name')
    # settings for training and testing data, which can be from different datasets
    # Here, using testing data to generate some temp results for observing variations in the results
    # for "dataset_name" (e.g. SSHR_SSHR), the first and second "SSHR"s refer to training and testing dataset name, respectively
    # this "dataset_name" can be used for mkdir specific dirs for saving results
    parser.add_argument('-dn', '--dataset_name', type=str, default='SSHR')
    parser.add_argument('-trdd', '--train_data_dir', type=str, default='dataset/shapenet_specular_1500/training_data')
    parser.add_argument('-trdlf', '--train_data_list_file', type=str, default='dataset/shapenet_specular_1500/train.lst')
    return parser

def train_model(UNet1, UNet2, UNet3, UNet4, dataloader, load_num_epoch, num_epoch, lr, dataset_name, parser, save_model_name='model'):

    # ensure dirs for saving results
    check_dir(dataset_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    UNet1.to(device)
    UNet2.to(device)
    UNet3.to(device)
    UNet4.to(device)

    # GPU in parallel
    if device == 'cuda':
        UNet1 = torch.nn.DataParallel(UNet1)
        UNet2 = torch.nn.DataParallel(UNet2)
        UNet3 = torch.nn.DataParallel(UNet3)
        UNet4 = torch.nn.DataParallel(UNet4)

    print("device:{}".format(device))

    beta1, beta2 = 0.5, 0.999

    optimizerUNet = torch.optim.Adam([{'params': UNet1.parameters()},
                                      {'params': UNet2.parameters()},
                                      {'params': UNet3.parameters()},
                                      {'params': UNet4.parameters()}],
                                     lr=lr, betas=(beta1, beta2))

    loss_criterion = nn.MSELoss().to(device)

    torch.backends.cudnn.benchmark = True

    mini_batch_size = parser.batch_size
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    UNets_loss = []

    if load_num_epoch is not None:
        epoch_old = int(load_num_epoch)
    else:
        epoch_old = 0

    for epoch in range(1, num_epoch+1):
        UNet1.train()
        UNet2.train()
        UNet3.train()
        UNet4.train()

        epoch = epoch + epoch_old
        epoch_L_total = 0.0

        print('Epoch {}/{}'.format(epoch, num_epoch+epoch_old))
        print('(train)')

        for img, gt_albedo, gt_shading, gt_specular_residue, gt_diffuse, gt_diffuse_tc, object_mask in tqdm(dataloader):
            img = img.to(device)
            gt_shading = gt_shading.to(device)
            gt_albedo = gt_albedo.to(device)
            gt_diffuse = gt_diffuse.to(device)
            gt_specular_residue = gt_specular_residue.to(device)
            gt_diffuse_tc = gt_diffuse_tc.to(device)

            ## estimations in our three-stage network
            # estimations in the first stage
            estimated_albedo = UNet1(img)
            estimated_shading = UNet2(img)
            estimated_specular_residue = (img - estimated_albedo * estimated_shading)

            # estimation in the second stage
            G3_input = torch.cat([estimated_albedo * estimated_shading, img], dim=1)
            estimated_diffuse_refined = UNet3(G3_input)

            # estimation in the third stage
            G4_input = torch.cat([estimated_diffuse_refined, estimated_specular_residue, img], dim=1)
            estimated_diffuse_tc = UNet4(G4_input)
            ## end

            # Train networks
            optimizerUNet.zero_grad()

            ## loss for our three-stage network

            # loss for the first stage (physics-based specular removal)
            L_albedo = loss_criterion(estimated_albedo, gt_albedo)
            L_shading = loss_criterion(estimated_shading, gt_shading)
            L_specular_residue = loss_criterion(estimated_specular_residue, gt_specular_residue)

            # loss for the second stage (specular-free refinement)
            L_diffuse_refined = loss_criterion(estimated_diffuse_refined, gt_diffuse)

            # loss for the thrid stage (tone correction)
            L_diffuse_tc = loss_criterion(estimated_diffuse_tc, gt_diffuse_tc)
            ## end

            # total loss
            # If you want to obtain better albedo and shading images,
            # it is better to use a lower weighting parameter for the loss of specular residue learning.
            # This is mainly attributed to the data transfer from high
            # dynamic range to low dynamic range (I is not equal to # A*S+R).
            # Please, uncomment the following first line and
            # comment the following second line
            # L_total = L_albedo +  L_shading + 0.01 * L_specular_residue + L_diffuse_refined + L_diffuse_tc
            L_total = L_albedo + L_shading + L_specular_residue + L_diffuse_refined + L_diffuse_tc

            L_total.backward()
            optimizerUNet.step()

            epoch_L_total += L_total.item()

        print('epoch {} || Epoch_Net_Loss:{:.4f}'.format(epoch, epoch_L_total/batch_size))

        UNets_loss += [epoch_L_total/batch_size]
        t_epoch_start = time.time()
        plot_log({'UNets':UNets_loss}, dataset_name, save_model_name)

        if(epoch%10 == 0):
            torch.save(UNet1.state_dict(), 'checkpoints_'+dataset_name+'/'+save_model_name+'1_'+str(epoch)+'.pth')
            torch.save(UNet2.state_dict(), 'checkpoints_'+dataset_name+'/'+save_model_name+'2_'+str(epoch)+'.pth')
            torch.save(UNet3.state_dict(), 'checkpoints_'+dataset_name+'/'+save_model_name+'3_'+str(epoch)+'.pth')
            torch.save(UNet4.state_dict(), 'checkpoints_'+dataset_name+'/'+save_model_name+'4_'+str(epoch)+'.pth')

            # update learning rate
            lr /= 10

    return UNet1, UNet2, UNet3, UNet4

def main(parser):
    UNet1 = UNet(input_channels=3, output_channels=3)
    UNet2 = UNet(input_channels=3, output_channels=3)
    UNet3 = UNet(input_channels=6, output_channels=3)
    UNet4 = UNet(input_channels=9, output_channels=3)

    mean = (0.5,)
    std = (0.5,)

    size = parser.image_size
    crop_size = parser.crop_size
    batch_size = parser.batch_size
    num_epoch = parser.num_epoch
    train_data_dir = parser.train_data_dir
    train_data_list_file = parser.train_data_list_file
    lr = parser.lr
    dataset_name = parser.dataset_name
    pretrained_dataset_name = parser.pretrained_dataset_name
    load_num_epoch =parser.load_num_epoch

    if parser.load_num_epoch is not None:
        print('load checkpoint ' + parser.load_num_epoch)
        # load UNet weights
        UNet1_weights = torch.load('./checkpoints_'+pretrained_dataset_name+'/UNet1_'+parser.load_num_epoch+'.pth')
        UNet1.load_state_dict(fix_model_state_dict(UNet1_weights))
        UNet2_weights = torch.load('./checkpoints_'+pretrained_dataset_name+'/UNet2_'+parser.load_num_epoch+'.pth')
        UNet2.load_state_dict(fix_model_state_dict(UNet2_weights))
        UNet3_weights = torch.load('./checkpoints_'+pretrained_dataset_name+'/UNet3_'+parser.load_num_epoch+'.pth')
        UNet3.load_state_dict(fix_model_state_dict(UNet3_weights))
        UNet4_weights = torch.load('./checkpoints_'+pretrained_dataset_name+'/UNet4_'+parser.load_num_epoch+'.pth')
        UNet4.load_state_dict(fix_model_state_dict(UNet4_weights))

    train_img_list = generate_training_data_list(train_data_dir, train_data_list_file)
    train_dataset = ImageDataset(img_list=train_img_list,
                                img_transform=ImageTransform(size=size, mean=mean, std=std),
                                phase='train')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    UNet1_ud, UNet2_ud, UNet3_ud, UNet4_ud = train_model(UNet1, UNet2, UNet3, UNet4,
                                                         dataloader=train_dataloader,
                                                         load_num_epoch=load_num_epoch,
                                                         num_epoch=num_epoch,
                                                         lr=lr, dataset_name=dataset_name,
                                                         parser=parser, save_model_name='UNet')


if __name__ == "__main__":
    parser = get_parser().parse_args()
    main(parser)
