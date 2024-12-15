import numpy as np
import os
from PIL import Image

subject_data = np.load('data/processed_data/subj01/nsd_test_stim_sub1.npy')

stimuli_output_dir = 'data/nsddata_stimuli/test_images/'

if not os.path.exists(stimuli_output_dir):
    os.makedirs(stimuli_output_dir)

for idx in range(len(subject_data)):
    image = Image.fromarray(subject_data[idx].astype(np.uint8))
    image.save('{}/{}.png'.format(stimuli_output_dir, idx))
