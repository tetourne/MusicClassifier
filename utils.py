import os
import pandas as pd
import ast


def load(filepath: str) -> pd.DataFrame:
    """Load CSV file, base on filename.

    Args:
        Symbol’s function definition is void: python-args-at-point

    Returns:
        The DataFrame containing the CSV data.
    """
    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks

def get_audio_path(audio_dir: str, track_id: int) -> str:
    """Return the path to the audio file given the directory
    where the audio is stored and the track ID.

    Args:
        audio_dir: The root directory of all audio files.
        track_id: The track ID of the file to retrieve.

    Returns:
        The path to the audio files.
    """
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

def get_audios(track_ids: pd.Index) -> pd.DataFrame:
    """Load all the audio files corresponding to the given track IDs and
    store them in a DataFrame.

    Args:
        track_ids: The list of all track IDs to read.

    Returns:
        The DataFrame containing the data.
    Raises:
        FileNotFoundError: If an audio file for one of the track IDs
        cannot be found.
    """
    audios = []
    samplerates = []
    for i in tqdm(track_ids):
        filename = utils.get_audio_path(audio_dir, i)
        x, sr = librosa.load(filename, sr=None, mono=True)
        audios.append(x)
        samplerates.append(sr)

    data = {'audio': audios,
            'sample rate': samplerates}
    X_train = pd.DataFrame(data, index=track_ids)
    return X_train
