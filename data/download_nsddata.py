import os
import boto3

# Download Experiment Infos
os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat nsddata/experiments/nsd/')
os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.pkl nsddata/experiments/nsd/')

# Download Stimuli
os.system('aws s3 cp s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 nsddata_stimuli/stimuli/nsd/')

# Download Betas
for sub in [1,2,5,7]:
    for sess in range(1,38):
        os.system('aws s3 cp s3://natural-scenes-dataset/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{:02d}.nii.gz nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub,sess,sub))

# # Download ROIs
# for sub in [1,2,5,7]:
#     os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/* nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub,sub))

s3 = boto3.client('s3')
def download_s3_folder(bucket_name, s3_folder, local_path):
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        if 'Contents' in page:
            for obj in page['Contents']:
                s3_file_path = obj['Key']
                local_file_path = os.path.join(local_path, os.path.relpath(s3_file_path, s3_folder))
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                s3.download_file(bucket_name, s3_file_path, local_file_path)

bucket_name = 'natural-scenes-dataset'
for sub in [1, 2, 5, 7]:
    s3_folder = f'nsddata/ppdata/subj{sub:02d}/func1pt8mm/roi/'
    local_path = f'nsddata/ppdata/subj{sub:02d}/func1pt8mm/roi/'
    download_s3_folder(bucket_name, s3_folder, local_path)