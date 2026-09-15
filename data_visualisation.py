import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import sklearn as skl

# Loading data
data_dir = '/run/media/thomas/Data/Documents/Database'
print(f"Reading data from {data_dir}...")
audio_dir = data_dir+'/fma_small'
tracks = utils.load(data_dir+'/fma_metadata/tracks.csv')
genres = utils.load(data_dir+'/fma_metadata/genres.csv')
features = utils.load(data_dir+'/fma_metadata/features.csv')
echonest = utils.load(data_dir+'/fma_metadata/echonest.csv')
np.testing.assert_array_equal(features.index, tracks.index)
assert echonest.index.isin(tracks.index).all()
print("Data loaded correctly.")

# Selecting data sets
dataset_name = "small"
print(f"\n\nSelecting data set: {dataset_name}...")
smallset_ids = tracks['set', 'subset'] <= dataset_name
tracks = tracks[smallset_ids]
features = features[smallset_ids]
print("Done selecting.")

print("\n\nSelecting training, validating and test sets...")
trainingset_ids = tracks['set', 'split'] == 'training'
tracks = tracks[trainingset_ids]
features = features[trainingset_ids]
print("Done selecting.")


