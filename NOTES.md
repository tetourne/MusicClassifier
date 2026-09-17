# Description
A project that aims at classifying songs based upon their musical genres and generate playlists according to these genres.

## Ideas
- The idea is to first use FMA data to choose a model that can predict the genre from only the audio file.
- Compare CNN to classic ML algo
- Once a model perform ok on FMA data, try to increase the data set. Maybe with data from MusicBrainz. Or other data sets. **The key will be to have a data set large enough.**
- First work with one label per song. Then multi label.
- Try to classify from audio only. Then try to add features extracted from audio. Use librosa to extract those features. --> It is not possible to classify from audio only. There is way too many data to train on. It's like 1.3 million sample for one single 30 s audio file decoded at 44.1 kHz. The solution is to work with audio features.
- Add data from MTG Jamendo
- I need to compute audio features for all the audio, to reduce the number of data to train on.
- Use beat-sync features: instead of calculating features based on arbitrary time windows, we can align them with beats.
- Use harmonic vs percussive components. Called HPSS: Harmonic-Percussive Source Separation. Then it's possible to extract features separately from each rather than on the original audio.
- Use Mel spectrograms (frequency vs time images) and train CNN models on those.

## To do
- chech the features used in FMA.
- compute the base features: MFCC + spectral centroid + bandwidth + rolloff + contrast + ZCR + RMS. Then compare several models (SVM, RF, gradient boosting, ...).
- Then, with the best model, add sets of features: chroma, tempo + onset, delta MFCCs, tonnetz, and see how it affects performances. Eventually try with all features. This will tell which types of musical information actually contribute to genre classification?
- Perform ablation: check how performances change when feature X is removed. That can help to get rid of non significant features that are computationally expensive. This will give a better hint of usefulness of each feature.
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
The data is memory heavy. I would probably need to work on some online CPUs.
I can't train model directly on the audio files. There are way too many data to do so. I need to reduce the data. This is done by computing the features of all the audios. Then I can train the models on these features.


## IV) Preparing the data

## V) Model exploration

## VI) Fine-tuning models

## VII) Final solution

## VIII) Launching and releasing the solution


# Feature analysis
Questions:
- which features are temporal, which ones are based on frenquency?
- is it useful to include median, min or max in addition to mean and std for the features?
- Is it interesting to get different values of spectral rolloff for different percentage cutoff? What would be those interesting percentage values?
- I don't understand how spectral contrast works. How are the frequency bands defined? Maybe it's interresting to get different values of spectral contrast for different frequency bands?
- Is delta MFCCs the derivative of MFCCs?
- For all these features, is it computationnaly expensive?
- Which of the beat features are the most important? How do I compute them with librosa?
- More explanations about Onset rate? What is a musical event? A song? A bar? A musical section of the song?
- About point 26. Is it better to use SVM or is it better to check first what algorithm performs the best?

## List of all features to investigate
### Spectral centroid 
It's basicaly the center of gravity of the frequencies.
Frequency based.
Computationally cheap.

### Spectral bandwidth
It basicaly tells how spread out the frequencies are around the centroid.
Frequency based.
Computationally cheap.

### Spectral rolloff
It is the frequency below which a specified percentage of the energy lies.
It would be good to compute 3 different rolloff frequencies with 3 different percentages: 50% (half), 85% (default), 95% (high frequency tail).
It would be interesting to compute rolloff_95 - rolloff_50 to see how broad the upper-frequency distribution is.	
Frequency based.
Computationally cheap.
This is an interesting feature.

### Spectral contrast
It attempts to characterize the difference between peaks and valleys of energy across frequency bands.
Librosa's default is roughly 7 frequency bands, starting from a minimum frequency and expanding upward approximately by octaves.
Frequency based.
Computationally cheap/moderate.
This one is very interesting.

### Spectral flatness
It measures how noise-like versus tone-like the spectrum is.
Frequency based.
Computationally cheap.
Less interesting but good to keep I guess.

### Spectral flux / spectral change
It tells how quickly is the spectrum changing.
Librosa doesn't have a function for that. But possible to compute from successive STFT frames.
Frequency based.

### Zero-crossing rate
It measures how frequently the waveform crosses zero.
Time/amplitude based.
Computationally cheap.

### RMS energy
It is a measure of the signal's average energy/loudness over a short period.
Nice to have mean, std, max.
Time/amplitude based.
Computationally cheap.

### MFCCs
MFCC = Mel-Frequency Cepstral Coefficients.
They're widely used in speech and audio classification.
The underlying idea is complicated, but the intuition is relatively simple:
MFCCs provide a compact representation of the spectral shape/timbre of an audio signal, using a frequency scale designed to roughly reflect human hearing.
The process is roughly: 
Audio
  ↓
STFT
  ↓
Power spectrum
  ↓
Mel filter bank
  ↓
log
  ↓
DCT
  ↓
MFCCs

MFCCs capture characteristics of timbre.
Frequency/timbre based.
Computationally moderate.
MFCCs would probably be the most important feature family.


### Delta and delta-delta MFCCs
It is basicaly how MFCCs are changing (delta) and how this delta is changing (delta-delta). So basicaly the derivative and second derivative of MFCCs.
Frequency/timbre based.
Computationally moderate.
Also important.

### Chroma
It captures information about harmonic content.
Frequency/pitch based.
Computationally moderate.
This is important.

### Tonnetz 
A more specialized harmonic representation.
Related to chroma but encodes harmonic relationships differently.
Frequency/harmony based.
Computationally moderate.
It is an advanced feature but can be interesting.

### Tempo
Beats per second of the track.
Time/rhythm based.

### Beat / rhythm features
We can also look at:
- beat intervals
- onset density
- rhythmic regularity
- beat strength
- rhythmic patterns
The most important ones (or the first to check) would be tempo, beat positions and onset strength.
From beat positions, we can look at the regularity of beat intervals for instance (with std  of beat intervals for instance).
For onset strenght, look at mean and std at first.
Time/rhythm based.


### Onset rate
It is essentially the beginning of a musical event. A musical event is a sound event, so it can be a note, a drum hit, a vocal syllabe etc.
So the onset rate is how frequently the audio contains detected new musical events.
To use with caution though, because the detection parameters of these events will strongly affect the result.
Time/rhythm based.
It is useful for differenciate rhythmically dense genres.

### Loudness and dynamics
Worth investigating:
- peak amplitude
- crest factor
- dynamic range
- loudness
- loudness variation
Time based.
These features could be interesting for the analysis, although beware:
Production/mixing/mastering can become a shortcut for genre.

### Spectral statistics
For instances:
- spectral skewness: measures the asymmetry of the frequency distribution.
- spectral kurtosis: measures how concentrated or extreme the distribution is.
Frequency based.
Maybe not as important.


## Feature families
- Timbre: MFCC
- Spectral shape: centroid, bandwidth, rolloff, contrast, flatness
- Energy: RMS
- Noisiness/percussiveness: ZCR
- Harmony: chroma, Tonnetz
- Rhythm: tempo, onset features
- Temporal dynamics: delta MFCC, delta-delta

See how each family affects the performances of the classifier.


# Reports and time spent
### 15/09/2026
Approx 5 hours.
Creating of GitHub repo. Retrieving of data. First look at data

### 16/09/2026
Approx 4 hours.
Understanding how to load the audio files. Spend some time to configure emacs correctly. Checked what docstrings to use.

### 17/09/2026
Approx 5 hours.
I've been taking some time to properly read the data. I've spent time to read and study about feature engineering, what features are important, how to compute them etc.
