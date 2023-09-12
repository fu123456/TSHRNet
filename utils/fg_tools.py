import os
import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict


def plot_log(data, dataset_name, save_model_name='model'):
    plt.cla()
    plt.plot(data['UNets'], label='L_total ')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.savefig('./logs_'+dataset_name+'/'+save_model_name+'.png')


def check_dir(dataset_name):
    if not os.path.exists('./logs_' + dataset_name):
        os.mkdir('./logs_' + dataset_name)
    if not os.path.exists('./checkpoints_' + dataset_name):
        os.mkdir('./checkpoints_' + dataset_name)


def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor((0.5, )) + torch.Tensor((0.5, ))
    x = x.transpose(1, 3)
    return x


def evaluate(UNet1, UNet2, UNet3, UNet4, dataset, device, filename):
    img, gt_albedo, gt_shading, gt_specular_residue, gt_diffuse, gt_diffuse_tc, object_mask = zip(*[dataset[i] for i in range(8)])
    img = torch.stack(img)
    gt_diffuse = torch.stack(gt_diffuse)
    gt_diffuse_tc = torch.stack(gt_diffuse_tc)
    object_mask = torch.stack(object_mask)


    img = img.to(device)
    object_mask = object_mask.to(device)

    with torch.no_grad():
        ## estimations in our three-stage network
        # estimations in the first stage
        estimated_albedo = UNet1(img)
        estimated_shading = UNet2(img)
        estimated_specular_residue = (img - estimated_albedo * estimated_shading)

        # estimation in the second stage
        G3_input = torch.cat([estimated_albedo * estimated_shading * object_mask, img], dim=1)
        estimated_diffuse_refined = UNet3(G3_input)

        # estimation in the third stage
        G4_input = torch.cat([estimated_diffuse_refined * object_mask, estimated_specular_residue * object_mask, img], dim=1)
        estimated_diffuse_tc = UNet4(G4_input)
        ## end

        # to cpu
        estimated_albedo = estimated_albedo.to(torch.device('cpu'))
        estimated_shading = estimated_shading.to(torch.device('cpu'))
        estimated_diffuse_refined = estimated_diffuse_refined.to(torch.device('cpu'))
        estimated_diffuse_tc = estimated_diffuse_tc.to(torch.device('cpu'))
        img = img.to(torch.device('cpu'))
        object_mask = object_mask.to(torch.device('cpu'))

    grid_removal = make_grid(torch.cat((img,gt_diffuse,estimated_diffuse_refined * object_mask, estimated_diffuse_tc * object_mask), dim=0))
    save_image(grid_removal, filename+'_overview.jpg')


def evaluate_mix(UNet1, UNet2, UNet3, UNet4, dataset, device, filename):
    input_img, gt_specular_residue, gt_diffuse, gt_diffuse_tc = zip(*[dataset[i] for i in range(16)]) # 8 in default
    input_img = torch.stack(input_img)
    gt_diffuse = torch.stack(gt_diffuse)
    gt_diffuse_tc = torch.stack(gt_diffuse_tc)

    with torch.no_grad():
        # first stage (physics-based specular highlight removal)
        estimated_diffuse = UNet1(input_img.to(device))
        estimated_specular_residue = UNet2(input_img.to(device))

        # second stage (specular-free refinement)
        G3_data = torch.cat([estimated_diffuse, input_img.to(device)], dim=1)
        estimated_diffuse_refined = UNet3(G3_data.to(device))

        # third stage (tone correction)
        input_img = input_img.to(device)
        G4_input = torch.cat([estimated_diffuse_refined, estimated_specular_residue, input_img], dim=1)
        estimated_diffuse_tc = UNet4(G4_input.to(device))

        # to cpu
        estimated_diffuse = estimated_diffuse.to(torch.device('cpu'))
        estimated_specular_residue = estimated_specular_residue.to(torch.device('cpu'))
        estimated_diffuse_refined = estimated_diffuse_refined.to(torch.device('cpu'))
        estimated_diffuse_tc = estimated_diffuse_tc.to(torch.device('cpu'))
        input_img = input_img.to(torch.device('cpu'))

    grid_removal = make_grid(torch.cat((unnormalize(input_img), unnormalize(gt_diffuse), unnormalize(estimated_diffuse_refined), unnormalize(estimated_diffuse_tc)), dim=0))
    save_image(grid_removal, filename+'_overview.jpg')
