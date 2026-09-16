# Description
A project that aims at classifying songs based upon their musical genres and generate playlists according to these genres.

## Ideas
- The idea is to first use FMA data to choose a model that can predict the genre from only the audio file.
- Compare CNN to classic ML algo
- Once a model perform ok on FMA data, try to increase the data set. Maybe with data from MusicBrainz. Or other data sets. **The key will be to have a data set large enough.**
- 



# Road map
## I) Framing the problem and big picture
### Define the objective
Classify songs based on their musical genres and eventually sort the songs and generate playlists based on the genres.
Eventualy, playlist could be generated using mood or emotions.
### How the solution is used
Preliminary. A python code is run on a music library, sets the tag for musical genre on all the files. An alternative tool is used to generate playlists.
Ultimately. Online tool where people can upload their library, get their songs classified and organised into an easy and handy architecture. The online tool will include options to generate playlists, based on requirements. Eventually, it will generate playlist based on mood or emotions.
### Current solutions
- https://picard.musicbrainz.org/
MusicBrainz Picard is a free, open-source music tagging application that identifies, organizes, and updates metadata for digital audio files. It is the official tagger for the MusicBrainz music database and is available for Windows, macOS, and Linux. Its combination of community-maintained metadata and acoustic fingerprinting makes it one of the most powerful tools for cleaning and organizing large music collections. 
- https://essentia.upf.edu/
Essentia is an open-source software library for audio analysis and music information retrieval (MIR). Originally developed by the Music Technology Group at Universitat Pompeu Fabra, it has become a widely used toolkit in both academic research and industry for extracting musical features, analyzing recordings, and building music-aware applications. 
### Problem framing
This problem is a multioutput classification problem. It is multiclass and multilabel. 
For now it's going to be an offline supervised model. Maybe it will involve into a online supervised model. 
We are going to use simple models at first such as random forest. And then eventually try more complexe models.
Could be interesting do see how CNN performs.
### Performances
The performances of the classifier are measured using precision and recall.
precision = TP / (TP + FP)
recall = TP / (TP + FN)
Both are combined into F1 score.
We will most likely focus on a good precision rather than a good recall.

## II) Getting the data
We start with a small sample that is FMA_small.
FMA is an open and easily accessible dataset suitable for evaluating several tasks in MIR, a field concerned with browsing, searching, and organizing large music collections. The community's growing interest in feature and end-to-end learning is however restrained by the limited availability of large audio datasets. The FMA aims to overcome this hurdle by providing 917 GiB and 343 days of Creative Commons-licensed audio from 106,574 tracks from 16,341 artists and 14,854 albums, arranged in a hierarchical taxonomy of 161 genres. It provides full-length and high-quality audio, pre-computed features, together with track- and user-level metadata, tags, and free-form text such as biographies. We here describe the dataset and how it was created, propose a train/validation/test split and three subsets, discuss some suitable MIR tasks, and evaluate some baselines for genre recognition. Code, data, and usage examples are available at https://github.com/mdeff/fma.

FMA_small is: 8,000 tracks of 30s, 8 balanced genres (GTZAN-like) (7.2 GiB).

## III) Exploring the data and first insights

## IV) Preparing the data

## V) Model exploration

## VI) Fine-tuning models

## VII) Final solution

## VIII) Launching and releasing the solution





# Reports and time spent
### 15/09/2026
Approx 5 hours.
Creating of GitHub repo. Retrieving of data. First look at data

### 16/09/2026
