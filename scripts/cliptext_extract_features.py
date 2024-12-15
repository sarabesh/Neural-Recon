import sys
sys.path.append('versatile_diffusion')
import os
import numpy as np

import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import matplotlib.pyplot as plt
import torchvision.transforms as T

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
subject_id = int(args.sub)
assert subject_id in [1,2,5,7]

config_model_name = 'vd_noema'
model_path = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
model_config = model_cfg_bank()(config_model_name)
model_network = get_model()(model_config)
state_dict = torch.load(model_path, map_location='cpu')
model_network.load_state_dict(state_dict, strict=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_network.clip = model_network.clip.to(device)

train_annotations = np.load('data/processed_data/subj{:02d}/nsd_train_cap_sub{}.npy'.format(subject_id, subject_id)) 
test_annotations = np.load('data/processed_data/subj{:02d}/nsd_test_cap_sub{}.npy'.format(subject_id, subject_id))

embedding_dim, feature_size, num_test_samples, num_train_samples = 77, 768, len(test_annotations), len(train_annotations)

train_embeddings = np.zeros((num_train_samples, embedding_dim, feature_size))
test_embeddings = np.zeros((num_test_samples, embedding_dim, feature_size))

with torch.no_grad():
    for idx, annotations in enumerate(test_annotations):
        cleaned_annotations = list(annotations[annotations != ''])
        print(idx)
        encoded_features = model_network.clip_encode_text(cleaned_annotations)
        test_embeddings[idx] = encoded_features.to('cpu').numpy().mean(0)

    np.save('data/extracted_features/subj{:02d}/nsd_cliptext_test.npy'.format(subject_id), test_embeddings)

    for idx, annotations in enumerate(train_annotations):
        cleaned_annotations = list(annotations[annotations != ''])
        print(idx)
        encoded_features = model_network.clip_encode_text(cleaned_annotations)
        train_embeddings[idx] = encoded_features.to('cpu').numpy().mean(0)

    np.save('data/extracted_features/subj{:02d}/nsd_cliptext_train.npy'.format(subject_id), train_embeddings)
