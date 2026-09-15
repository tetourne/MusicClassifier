import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import sklearn as skl

# Loading data
data_dir = '/run/media/thomas/Data/Documents/Database'
audio_dir = data_dir+'/fma_small'
tracks = utils.load(data_dir+'/fma_metadata/tracks.csv')
genres = utils.load(data_dir+'/fma_metadata/genres.csv')
features = utils.load(data_dir+'/fma_metadata/features.csv')
echonest = utils.load(data_dir+'/fma_metadata/echonest.csv')
np.testing.assert_array_equal(features.index, tracks.index)
assert echonest.index.isin(tracks.index).all()

# Selecting data sets
msk = [tracks['set', 'subset'] <= 'small']
tracks = tracks[msk]

# First insights
print(tracks.head())
print(tracks.info())
print(tracks.describe())
