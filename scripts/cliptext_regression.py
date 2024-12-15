import sys
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1)
args = parser.parse_args()
subject = int(args.sub)
assert subject in [1, 2, 5, 7]

train_path = 'data/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(subject, subject)
train_fmri_data = np.load(train_path)
test_path = 'data/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(subject, subject)
test_fmri_data = np.load(test_path)

train_fmri_data = train_fmri_data / 300
test_fmri_data = test_fmri_data / 300

mean_train_data = np.mean(train_fmri_data, axis=0)
std_train_data = np.std(train_fmri_data, axis=0, ddof=1)
train_fmri_data = (train_fmri_data - mean_train_data) / std_train_data
test_fmri_data = (test_fmri_data - mean_train_data) / std_train_data

print(np.mean(train_fmri_data), np.std(train_fmri_data))
print(np.mean(test_fmri_data), np.std(test_fmri_data))

print(np.max(train_fmri_data), np.min(train_fmri_data))
print(np.max(test_fmri_data), np.min(test_fmri_data))

num_voxels, num_train_samples, num_test_samples = train_fmri_data.shape[1], len(train_fmri_data), len(test_fmri_data)

train_latent_features = np.load('data/extracted_features/subj{:02d}/nsd_cliptext_train.npy'.format(subject))
test_latent_features = np.load('data/extracted_features/subj{:02d}/nsd_cliptext_test.npy'.format(subject))

num_latents, num_embeddings, num_dimensions = train_latent_features.shape

print("Training Regression Model")
weights_matrix = np.zeros((num_embeddings, num_dimensions, num_voxels)).astype(np.float32)
biases_vector = np.zeros((num_embeddings, num_dimensions)).astype(np.float32)
predicted_latent = np.zeros_like(test_latent_features)

for idx in range(num_embeddings):
    ridge_reg = skl.Ridge(alpha=100000, max_iter=50000, fit_intercept=True)
    ridge_reg.fit(train_fmri_data, train_latent_features[:, idx])
    weights_matrix[idx] = ridge_reg.coef_
    biases_vector[idx] = ridge_reg.intercept_

    predictions = ridge_reg.predict(test_fmri_data)
    normalized_predictions = (predictions - np.mean(predictions, axis=0)) / np.std(predictions, axis=0)
    predicted_latent[:, idx] = normalized_predictions * np.std(train_latent_features[:, idx], axis=0) + np.mean(train_latent_features[:, idx], axis=0)
    print(idx, ridge_reg.score(test_fmri_data, test_latent_features[:, idx]))

np.save('data/predicted_features/subj{:02d}/nsd_cliptext_predtest_nsdgeneral.npy'.format(subject), predicted_latent)

regression_results = {
    'weights': weights_matrix,
    'biases': biases_vector,
}

with open('data/regression_weights/subj{:02d}/cliptext_regression_weights.pkl'.format(subject), "wb") as save_file:
    pickle.dump(regression_results, save_file)
