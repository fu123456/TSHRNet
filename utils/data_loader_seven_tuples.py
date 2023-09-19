import os
import torch.utils.data as data
from . import seven_tuple_data_processing
from PIL import Image
import random
from torchvision import transforms


def generate_training_data_list(training_data_dir, training_data_list_file):
    # shapenet_specular training dataset
    # training_data_dir = 'dataset/shapenet_specular_1500/training_data'
    # training_data_list_file = 'dataset/shapenet_specular_1500/train_tc.lst'

    random.seed(1)

    path_i = [] # input
    path_a = [] # albedo
    path_s = [] # shading
    path_r = [] # specular residue
    path_d = [] # diffuse
    path_d_tc = [] # gamma correction version of diffuse
    path_m = [] # mask
    with open(training_data_list_file, 'r') as f:
        image_list = [x.strip() for x in f.readlines()]
    random.shuffle(image_list)
    for name in image_list:
        path_i.append(os.path.join(training_data_dir, name.split()[0])) # input
        path_a.append(os.path.join(training_data_dir, name.split()[1])) # albedo
        path_s.append(os.path.join(training_data_dir, name.split()[2])) # shading
        path_r.append(os.path.join(training_data_dir, name.split()[3])) # specular residue
        path_d.append(os.path.join(training_data_dir, name.split()[4])) # diffuse
        path_d_tc.append(os.path.join(training_data_dir, name.split()[5])) # gamma correction version of diffuse
        path_m.append(os.path.join(training_data_dir, name.split()[6])) # mask

    num = len(image_list)
    path_i = path_i[:int(num)]
    path_a = path_a[:int(num)]
    path_s = path_s[:int(num)]
    path_r = path_r[:int(num)]
    path_d = path_d[:int(num)]
    path_d_tc = path_d_tc[:int(num)]
    path_m = path_m[:int(num)]

    path_list = {'path_i': path_i, 'path_a': path_a, 'path_s': path_s, 'path_r': path_r, 'path_d': path_d, 'path_d_tc': path_d_tc, 'path_m': path_m}
    return path_list


def generate_testing_data_list(data_dir, data_list_file):
    # shapenet_specular testing data
    # data_dir = 'dataset/shapenet_specular_1500/testing_data'
    # data_list_file = 'dataset/shapenet_specular_1500/test_tc.lst'

    path_i = [] # input
    path_a = [] # albedo
    path_s = [] # shading
    path_r = [] # specular residue
    path_d = [] # diffuse
    path_d_tc = [] # gamma correction version of diffuse
    path_m = [] # mask
    with open(data_list_file, 'r') as f:
        image_list = [x.strip() for x in f.readlines()]
    image_list.sort()
    for name in image_list:
        path_i.append(os.path.join(data_dir, name.split()[0])) # input
        path_a.append(os.path.join(data_dir, name.split()[1])) # albedo
        path_s.append(os.path.join(data_dir, name.split()[2])) # shading
        path_r.append(os.path.join(data_dir, name.split()[3])) # specular residue
        path_d.append(os.path.join(data_dir, name.split()[4])) # diffuse
        path_d_tc.append(os.path.join(data_dir, name.split()[5])) # gamma correction version of diffuse
        path_m.append(os.path.join(data_dir, name.split()[6])) # mask

    num = len(image_list)
    path_i = path_i[:int(num)]
    path_a = path_a[:int(num)]
    path_s = path_s[:int(num)]
    path_r = path_r[:int(num)]
    path_d = path_d[:int(num)]
    path_d_tc = path_d_tc[:int(num)]
    path_m = path_m[:int(num)]

    path_list = {'path_i': path_i, 'path_a': path_a, 'path_s': path_s, 'path_r': path_r, 'path_d': path_d, 'path_d_tc': path_d_tc, 'path_m': path_m}

    return path_list


class ImageTransformSingle():
    def __init__(self, size=256, mean=(0.5, ), std=(0.5, )):
        self.data_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])

    def __call__(self, img):
        return self.data_transform(img)


class ImageTransform():
    def __init__(self, size=256, crop_size=256, mean=(0.5, ), std=(0.5, )):
        self.data_transform = {'train': seven_tuple_data_processing.Compose([seven_tuple_data_processing.Scale(size=size),
                                                                             seven_tuple_data_processing.ToTensor(),
                                                                             seven_tuple_data_processing.Normalize(mean, std)]),

                                'test': seven_tuple_data_processing.Compose([seven_tuple_data_processing.Scale(size=size),
                                                                             seven_tuple_data_processing.ToTensor(),
                                                                             seven_tuple_data_processing.Normalize(mean, std)])}

    def __call__(self, phase, img):
        return self.data_transform[phase](img)


class ImageDataset(data.Dataset):
    def __init__(self, img_list, img_transform, phase):
        self.img_list = img_list
        self.img_transform = img_transform
        self.phase = phase

    def __len__(self):
        return len(self.img_list['path_i'])

    def __getitem__(self, index):
        input = Image.open(self.img_list['path_i'][index]).convert('RGB')
        gt_albedo = Image.open(self.img_list['path_a'][index]).convert('RGB')
        gt_shading = Image.open(self.img_list['path_s'][index]).convert('RGB')
        gt_specular_residue = Image.open(self.img_list['path_r'][index]).convert('RGB')
        gt_diffuse = Image.open(self.img_list['path_d'][index]).convert('RGB')
        gt_diffuse_tc= Image.open(self.img_list['path_d_tc'][index]).convert('RGB')
        object_mask= Image.open(self.img_list['path_m'][index]).convert('RGB')

        # data pre-processing
        input, gt_albedo, gt_shading, gt_specular_residue, gt_diffuse, gt_diffuse_tc, object_mask = self.img_transform(self.phase, [input, gt_albedo, gt_shading, gt_specular_residue, gt_diffuse, gt_diffuse_tc, object_mask])

        return input, gt_albedo, gt_shading, gt_specular_residue, gt_diffuse, gt_diffuse_tc, object_mask
