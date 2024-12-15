import sys
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1)
args = parser.parse_args()

subject_id = int(args.sub)
assert subject_id in [1, 2, 5, 7]

train_data_path = 'data/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(subject_id, subject_id)
train_fmri_data = np.load(train_data_path)

test_data_path = 'data/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(subject_id, subject_id)
test_fmri_data = np.load(test_data_path)

train_fmri_data = train_fmri_data / 300
test_fmri_data = test_fmri_data / 300

mean_train_fmri = np.mean(train_fmri_data, axis=0)
std_train_fmri = np.std(train_fmri_data, axis=0, ddof=1)
train_fmri_data = (train_fmri_data - mean_train_fmri) / std_train_fmri
test_fmri_data = (test_fmri_data - mean_train_fmri) / std_train_fmri

print(np.mean(train_fmri_data), np.std(train_fmri_data))
print(np.mean(test_fmri_data), np.std(test_fmri_data))

print(np.max(train_fmri_data), np.min(train_fmri_data))
print(np.max(test_fmri_data), np.min(test_fmri_data))

num_voxels, num_train_samples, num_test_samples = train_fmri_data.shape[1], len(train_fmri_data), len(test_fmri_data)

train_features = np.load('data/extracted_features/subj{:02d}/nsd_clipvision_train.npy'.format(subject_id))
test_features = np.load('data/extracted_features/subj{:02d}/nsd_clipvision_test.npy'.format(subject_id))

num_train_embed, num_embed_dim, num_feature_dim = train_features.shape

print("Training Ridge Regression")
weights = np.zeros((num_embed_dim, num_feature_dim, num_voxels)).astype(np.float32)
biases = np.zeros((num_embed_dim, num_feature_dim)).astype(np.float32)
predicted_features = np.zeros_like(test_features)

for idx in range(num_embed_dim):
    ridge_reg = skl.Ridge(alpha=60000, max_iter=50000, fit_intercept=True)
    ridge_reg.fit(train_fmri_data, train_features[:, idx])
    weights[idx] = ridge_reg.coef_
    biases[idx] = ridge_reg.intercept_

    test_predictions = ridge_reg.predict(test_fmri_data)
    normalized_test_predictions = (test_predictions - np.mean(test_predictions, axis=0)) / np.std(test_predictions, axis=0)
    predicted_features[:, idx] = normalized_test_predictions * np.std(train_features[:, idx], axis=0) + np.mean(train_features[:, idx], axis=0)

    print(idx, ridge_reg.score(test_fmri_data, test_features[:, idx]))

np.save('data/predicted_features/subj{:02d}/nsd_clipvision_predtest_nsdgeneral.npy'.format(subject_id), predicted_features)

regression_data = {
    'weights': weights,
    'biases': biases
}

with open('data/regression_weights/subj{:02d}/clipvision_regression_weights.pkl'.format(subject_id), "wb") as save_file:
    pickle.dump(regression_data, save_file)
