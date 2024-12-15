import sys
sys.path.append('versatile_diffusion')
import os
import PIL
from PIL import Image
import numpy as np

import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import torchvision.transforms as T

import argparse

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1)
args = parser.parse_args()
subject_number = int(args.sub)
assert subject_number in [1, 2, 5, 7]

cfgm_name = 'vd_noema'
pth_path = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfg_model = model_cfg_bank()(cfgm_name)
model_net = get_model()(cfg_model)
state_dict = torch.load(pth_path, map_location='cpu')
model_net.load_state_dict(state_dict, strict=False)

device_config = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_net.clip = model_net.clip.to(device_config)

class ExternalImageBatchGenerator(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images = np.load(dataset_path).astype(np.uint8)

    def __getitem__(self, index):
        img_data = Image.fromarray(self.images[index])
        img_data = T.functional.resize(img_data, (512, 512))
        img_data = T.functional.to_tensor(img_data).float()
        img_data = img_data * 2 - 1
        return img_data

    def __len__(self):
        return len(self.images)

batch_size_value = 1

train_data_path = 'data/processed_data/subj{:02d}/nsd_train_stim_sub{}.npy'.format(subject_number, subject_number)
train_data_loader = ExternalImageBatchGenerator(dataset_path=train_data_path)

test_data_path = 'data/processed_data/subj{:02d}/nsd_test_stim_sub{}.npy'.format(subject_number, subject_number)
test_data_loader = ExternalImageBatchGenerator(dataset_path=test_data_path)

train_loader = DataLoader(train_data_loader, batch_size_value, shuffle=False)
test_loader = DataLoader(test_data_loader, batch_size_value, shuffle=False)

num_visuals, num_features_count, num_test_samples, num_train_samples = 257, 768, len(test_data_loader), len(train_data_loader)

train_visual_features = np.zeros((num_train_samples, num_visuals, num_features_count))
test_visual_features = np.zeros((num_test_samples, num_visuals, num_features_count))

with torch.no_grad():
    for idx, input_batch in enumerate(test_loader):
        print(idx)
        encoded_data = model_net.clip_encode_vision(input_batch)
        test_visual_features[idx] = encoded_data[0].cpu().numpy()

    np.save('data/extracted_features/subj{:02d}/nsd_clipvision_test.npy'.format(subject_number), test_visual_features)

    for idx, input_batch in enumerate(train_loader):
        print(idx)
        encoded_data = model_net.clip_encode_vision(input_batch)
        train_visual_features[idx] = encoded_data[0].cpu().numpy()

    np.save('data/extracted_features/subj{:02d}/nsd_clipvision_train.npy'.format(subject_number), train_visual_features)
