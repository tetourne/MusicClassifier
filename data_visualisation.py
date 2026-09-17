import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import sklearn as skl
import librosa


# Loading data
data_dir = '/run/media/thomas/Data/Documents/Database'
print(f"Reading data from {data_dir}...")

tracks = utils.load(data_dir+'/fma_metadata/tracks.csv')
genres = utils.load(data_dir+'/fma_metadata/genres.csv')
features = utils.load(data_dir+'/fma_metadata/features.csv')
echonest = utils.load(data_dir+'/fma_metadata/echonest.csv')
np.testing.assert_array_equal(features.index, tracks.index)
assert echonest.index.isin(tracks.index).all()
print("Data loaded correctly.")

# Selecting labels
small = tracks['set', 'subset'] <= 'small'
train = tracks['set', 'split'] == 'training'
val = tracks['set', 'split'] == 'validation'
test = tracks['set', 'split'] == 'test'

y_train = tracks.loc[small & train, ('track', 'genre_top')]
y_test = tracks.loc[small & test, ('track', 'genre_top')]

# Opening audio files
audio_dir = data_dir+'/fma_small'
limit = 1000
print(f"Reading {limit+int(limit/5)} audio files from: {audio_dir}...")
indices = y_train.index[:limit]
X_train = utils.get_audios(audio_dir, indices)
indices = y_test.index[:int(limit/5)]
X_test = utils.get_audios(audio_dir, indices)
y_train = y_train[:limit]
y_test = y_test[:int(limit/5)]

print('{} training examples, {} testing examples'.format(y_train.size, y_test.size))
print('{} features, {} classes'.format(X_train.shape[1], np.unique(y_train).size))
