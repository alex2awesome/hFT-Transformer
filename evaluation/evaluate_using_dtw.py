# pip install git+https://github.com/alex2awesome/djitw.git

import librosa
import djitw
import numpy as np
import pretty_midi
import scipy
import IPython
import os

# Audio/CQT parameters
FS = 22050
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


if __name__ == "__main__":
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
    print("Confidence score: {}".format(score))