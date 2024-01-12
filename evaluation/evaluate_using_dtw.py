# pip install git+https://github.com/alex2awesome/djitw.git

import librosa
import djitw
import numpy as np
import pretty_midi
import scipy
import IPython
import os
import sqlite3
import pandas as pd
import random
import datetime
import glob
from tqdm.auto import tqdm

# Audio/CQT parameters
FS = 22050.
NOTE_START = 36
N_NOTES = 48
HOP_LENGTH = 1024
# DTW parameters
GULLY = .96


def compute_cqt(audio_data):
    """ Compute the CQT and frame times for some audio data """
    # Compute CQT
    cqt = librosa.cqt(
        audio_data,
        sr=FS,
        fmin=librosa.midi_to_hz(NOTE_START),
        n_bins=N_NOTES,
        hop_length=HOP_LENGTH,
        tuning=0.
    )
    # Compute the time of each frame
    times = librosa.frames_to_time(
        np.arange(cqt.shape[1]),
        sr=FS,
        hop_length=HOP_LENGTH
    )
    # Compute log-amplitude
    cqt = librosa.amplitude_to_db(cqt, ref=cqt.max())
    # Normalize and return
    return librosa.util.normalize(cqt, norm=2).T, times


def load_and_run_dtw(audio_file=None, midi_file=None, audio_packet=None):
    # Load in the audio data
    if audio_packet is None:
        audio_data, _ = librosa.load(audio_file, sr=FS)
        audio_cqt, audio_times = compute_cqt(audio_data)
    else:
        audio_cqt, audio_times, audio_data = audio_packet

    midi_object = pretty_midi.PrettyMIDI(midi_file)
    midi_audio = midi_object.fluidsynth(fs=FS)
    midi_cqt, midi_times = compute_cqt(midi_audio)

    # Nearly all high-performing systems used cosine distance
    distance_matrix = scipy.spatial.distance.cdist(midi_cqt, audio_cqt, 'cosine')

    # Get lowest cost path
    p, q, score = djitw.dtw(
        distance_matrix,
        GULLY,  # The gully for all high-performing systems was near 1
        np.median(distance_matrix),  # The penalty was also near 1.0*median(distance_matrix)
        inplace=False
    )
    # Normalize by path length, normalize by distance matrix submatrix within path
    score = score / len(p)
    score = score / distance_matrix[p.min():p.max(), q.min():q.max()].mean()
    # Adjust the MIDI file
    midi_object.adjust_times(midi_times[p], audio_times[q])
    # Synthesize aligned MIDI
    midi_audio_aligned = midi_object.fluidsynth(fs=FS)
    # Adjust to the same size as audio
    if midi_audio_aligned.shape[0] > audio_data.shape[0]:
        midi_audio_aligned = midi_audio_aligned[:audio_data.shape[0]]
    else:
        trim_amount = audio_data.shape[0] - midi_audio_aligned.shape[0]
        midi_audio_aligned = np.append(
            midi_audio_aligned, np.zeros(trim_amount)
        )
    return score, midi_audio_aligned, (audio_cqt, audio_times, audio_data)


def match_audio_midi_file_lists(audio_files, midi_files):
    audio = pd.DataFrame(audio_files, columns=['audio_file'])
    midi = pd.DataFrame(midi_files, columns=['midi_file'])
    audio['key'] = audio['audio_file'].apply(os.path.basename).apply(lambda x: x.replace('.wav', ''))
    midi['key'] = midi['midi_file'].apply(os.path.basename).apply(lambda x: x.replace('_transcribed.mid', ''))
    matched_files = pd.merge(audio, midi, on='key', how='inner')
    return matched_files[['audio_file', 'midi_file']].apply(list, axis=1).tolist()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-audio_file', help='audio file', default=None)
    parser.add_argument('-midi_file', help='midi file', default=None)

    parser.add_argument('-audio_file_pattern', default=None, help='audio file')
    parser.add_argument('-midi_file_pattern', default=None, help='audio file')
    parser.add_argument('-output_sqlite_db', default=None, help='output file')
    parser.add_argument('-start_index', default=None, help='start index')
    parser.add_argument('-end_index', default=None, help='end index')

    args = parser.parse_args()

    conn = sqlite3.connect(args.output_sqlite_db)
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE IF NOT EXISTS scores 
        (   audio_file text KEY, 
            midi_file text KEY,
            timestamp real,
            score real
        )
    ''')

    if (args.audio_file is not None) and (args.midi_file is not None):
        audio_files_to_process = [args.audio_file]
        midi_files_to_process = [args.midi_file]
    else:
        audio_files_to_process = glob.glob(args.audio_file_pattern)
        midi_files_to_process = glob.glob(args.midi_file_pattern)

    matched_files = match_audio_midi_file_lists(audio_files_to_process, midi_files_to_process)
    random.shuffle(matched_files)
    if (args.start_index is not None) and (args.end_index is not None):
        matched_files = matched_files[args.start_index:args.end_index]

    for audio_file, midi_file in tqdm(matched_files):
        # check if we've already processed this file
        c.execute('SELECT * FROM scores WHERE audio_file=? AND midi_file=?', (audio_file, midi_file))
        if len(c.fetchall()) > 0:
            continue

        print(audio_file, midi_file)
        score, _, _ = load_and_run_dtw(audio_file, midi_file)
        c.execute('INSERT INTO scores VALUES (?, ?, ?, ?)', (audio_file, midi_file, str(datetime.datetime.now()), score))
        conn.commit()







"""
python evaluate_using_dtw.py \
    -audio_file_pattern=/mnt/data10/spangher/aira-dl/corpus/*/*.wav \
    -midi_file_pattern=/mnt/data10/spangher/aira-dl/data/2024-01-09__full-corpus__006-009-model/*.midi \
    -output_sqlite_db=/mnt/data10/spangher/aira-dl/data/2024-01-09__full-corpus__006-009-model/scores.db

python /mnt/data10/spangher/aira-dl/hFT-Transformer/evaluation/evaluate_using_dtw.py \
    -audio_file_pattern=/mnt/data10/spangher/aira-dl/corpus/*/*.wav \
    -midi_file_pattern=/mnt/data10/spangher/aira-dl/data/2024-01-09__full-corpus__006-009-model/*.mid \
    -output_sqlite_db=/nas/home/spangher/scores.db
    
python /mnt/data10/spangher/aira-dl/hFT-Transformer/evaluation/evaluate_using_dtw.py \
    -audio_file_pattern=/mnt/data10/spangher/aira-dl/hFT-Transformer/evaluation/glenn-gould-bach-data/*.wav \
    -midi_file_pattern=/mnt/data10/spangher/aira-dl/hFT-Transformer/evaluation/glenn-gould-bach-data/*/*.mid \
    -output_sqlite_db=/nas/home/spangher/bach-eval-scores.db
"""

"""
one doc sample:

# We'll use a real-world MIDI/audio pair
bach_dir = '/mnt/data10/spangher/aira-dl/data/test-bach-with-diff-models'
audio_file = os.path.join(
    bach_dir,
    'Johann_Sebastian_Bach_Glenn_Gould_-_The_Well-Tempered_Clavier_Book_1_Fugue_No._6_in_D_Minor_BWV_851.wav'
)
midi_file = os.path.join(bach_dir, 'bach__baseline-hft.midi')

score, midi_audio_aligned, audio_proc_packet = load_and_run_dtw(audio_file, midi_file)
print("Confidence score: {}".format(score))

midi_file = os.path.join(bach_dir, 'bach__aug-hft__003_004.midi')
score, midi_audio_aligned, _ = load_and_run_dtw(midi_file=midi_file, audio_packet=audio_proc_packet)
print("Confidence score: {}".format(score))

midi_file = os.path.join(bach_dir, 'bach__aug-hft__005_006.midi')
score, midi_audio_aligned, _ = load_and_run_dtw(audio_file, midi_file)
print("Confidence score: {}".format(score))

# We'll use a real-world MIDI/audio pair
midi_file = os.path.join(bach_dir, 'bach__aug-hft__006_009.midi')
score, midi_audio_aligned, _ = load_and_run_dtw(audio_file, midi_file)
print("Confidence score: {}".format(score))"""