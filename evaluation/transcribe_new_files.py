#! python
import os
import argparse
import pickle
import json
import sys
import glob
sys.path.append(os.getcwd())
from model import amt
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import random

try:
    from speechbrain.pretrained import SpectralMaskEnhancement
    model = SpectralMaskEnhancement.from_hparams("speechbrain/mtl-mimic-voicebank")
except:
    pass

def check_and_convert_mp3_to_wav(fname):
    wav_file = fname.replace('.mp3', '.wav')
    if not os.path.exists(wav_file):
        print('converting ' + fname + ' to .wav...')
        try:
            sound = AudioSegment.from_mp3(fname)
            sound.export(fname.replace('.mp3', '.wav'), format="wav")
        except CouldntDecodeError:
            print('failed to convert ' + fname)
            return None
    return wav_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # necessary arguments
    parser.add_argument('-input_dir_to_transcribe', default=None, help='file list')
    parser.add_argument('-input_file_to_transcribe', default=None, help='one file')
    parser.add_argument('-output_dir', help='output directory')
    parser.add_argument('-output_file', default=None, help='output file')
    parser.add_argument('-f_config', help='config json file', default='../corpus/config.json')
    parser.add_argument('-model_file', help='input model file', default='best_model.pkl')
    parser.add_argument('-start_index', help='start index', type=int, default=None)
    parser.add_argument('-end_index', help='end index', type=int, default=None)
    parser.add_argument('-skip_transcribe_mp3', action='store_true', default=False)
    # parameters
    parser.add_argument('-mode', help='mode to transcript (combination|single)', default='combination')
    parser.add_argument('-thred_mpe', help='threshold value for mpe detection', type=float, default=0.5)
    parser.add_argument('-thred_onset', help='threshold value for onset detection', type=float, default=0.5)
    parser.add_argument('-thred_offset', help='threshold value for offset detection', type=float, default=0.5)
    parser.add_argument('-n_stride', help='number of samples for offset', type=int, default=0)
    parser.add_argument('-ablation', help='ablation mode', action='store_true')
    args = parser.parse_args()

    assert (args.input_dir_to_transcribe is not None) or (args.input_file_to_transcribe is not None), "input file or directory is not specified"

    if args.input_dir_to_transcribe is not None:
        if not args.skip_transcribe_mp3:
            # list file
            a_mp3s = (
                    glob.glob(os.path.join(args.input_dir_to_transcribe, '*.mp3')) +
                    glob.glob(os.path.join(args.input_dir_to_transcribe, '*', '*.mp3'))
            )
            print(f'transcribing {len(a_mp3s)} files: [{str(a_mp3s)}]...')
            list(map(check_and_convert_mp3_to_wav, a_mp3s))

        a_list = (
            glob.glob(os.path.join(args.input_dir_to_transcribe, '*.wav')) +
            glob.glob(os.path.join(args.input_dir_to_transcribe, '*', '*.wav'))
        )
        if (args.start_index is not None) or (args.end_index is not None):
            if args.start_index is None:
                args.start_index = 0
            if args.end_index is None:
                args.end_index = len(a_list)
            a_list = a_list[args.start_index:args.end_index]
        # shuffle a_list
        random.shuffle(a_list)

    elif args.input_file_to_transcribe is not None:
        args.input_file_to_transcribe = check_and_convert_mp3_to_wav(args.input_file_to_transcribe)
        if args.input_file_to_transcribe is None:
            sys.exit()
        a_list = [args.input_file_to_transcribe]
        print(f'transcribing {str(a_list)} files...')

    # config file
    with open(args.f_config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # AMT class
    AMT = amt.AMT(config, args.model_file, verbose_flag=False)

    long_filename_counter = 0
    for fname in a_list:
        if args.output_file is not None:
            output_fname = args.output_file
        else:
            output_fname = fname.replace('.wav', '')
            if len(output_fname) > 200:
                output_fname = output_fname[:200] + f'_fnabbrev-{long_filename_counter}'
            output_fname += '_transcribed.mid'
            output_fname = os.path.join(args.output_dir, os.path.basename(output_fname))
            if os.path.exists(output_fname):
                continue

        print('[' + fname + ']')
        try:
            a_feature = AMT.wav2feature(fname)

            # transcript
            if args.n_stride > 0:
                output = AMT.transcript_stride(a_feature, args.n_stride, mode=args.mode, ablation_flag=args.ablation)
            else:
                output = AMT.transcript(a_feature, mode=args.mode, ablation_flag=args.ablation)
            (output_1st_onset, output_1st_offset, output_1st_mpe, output_1st_velocity,
             output_2nd_onset, output_2nd_offset, output_2nd_mpe, output_2nd_velocity) = output

            # note (mpe2note)
            a_note_1st_predict = AMT.mpe2note(
                a_onset=output_1st_onset,
                a_offset=output_1st_offset,
                a_mpe=output_1st_mpe,
                a_velocity=output_1st_velocity,
                thred_onset=args.thred_onset,
                thred_offset=args.thred_offset,
                thred_mpe=args.thred_mpe,
                mode_velocity='ignore_zero',
                mode_offset='shorter'
            )

            a_note_2nd_predict = AMT.mpe2note(
                a_onset=output_2nd_onset,
                a_offset=output_2nd_offset,
                a_mpe=output_2nd_mpe,
                a_velocity=output_2nd_velocity,
                thred_onset=args.thred_onset,
                thred_offset=args.thred_offset,
                thred_mpe=args.thred_mpe,
                mode_velocity='ignore_zero',
                mode_offset='shorter'
            )

            AMT.note2midi(a_note_2nd_predict, output_fname)
        except Exception as e:
            print(e)
            continue

    print('** done **')


"""
python evaluation/transcribe_new_files.py \
    -input_dir_to_transcribe ../data/test-album \
    -output_dir ../data/test-album-transcribed \
    -f_config corpus/MAESTRO-V3/dataset/config-orig.json \
    -model_file evaluation/checkpoint/MAESTRO-V3/model_016_003.pkl
    
    
## transcribe full corpus

python evaluation/transcribe_new_files.py \
    -input_dir_to_transcribe ../data/corpus \
    -output_dir ../data/2024-01-09__full-corpus__006-009-model \
    -f_config corpus/MAESTRO-V3/dataset/config-aug.json \
    -model_file evaluation/checkpoint/MAESTRO-V3/model-with-aug-data_006_009.pkl    
    
    
    
## evaluate test sample

python evaluation/transcribe_new_files.py \
    -input_file_to_transcribe ../data/Johann_Sebastian_Bach_Glenn_Gould_-_The_Well-Tempered_Clavier_Book_1_Fugue_No._6_in_D_Minor_BWV_851.mp3 \
    -output_file ../data/test-bach-with-diff-models/bach__baseline-hft.midi \
    -f_config corpus/MAESTRO-V3/dataset/config-orig.json \
    -model_file evaluation/checkpoint/MAESTRO-V3/model_016_003.pkl

python evaluation/transcribe_new_files.py \
    -input_file_to_transcribe ../data/Johann_Sebastian_Bach_Glenn_Gould_-_The_Well-Tempered_Clavier_Book_1_Fugue_No._6_in_D_Minor_BWV_851.mp3 \
    -output_file ../data/test-bach-with-diff-models/bach__aug-hft__003_004.midi \
    -f_config corpus/MAESTRO-V3/dataset/config-aug.json \
    -model_file evaluation/checkpoint/MAESTRO-V3/model-with-aug-data_003_004.pkl    

python evaluation/transcribe_new_files.py \
    -input_file_to_transcribe ../data/Johann_Sebastian_Bach_Glenn_Gould_-_The_Well-Tempered_Clavier_Book_1_Fugue_No._6_in_D_Minor_BWV_851.mp3 \
    -output_file ../data/test-bach-with-diff-models/bach__aug-hft__005_006.midi \
    -f_config corpus/MAESTRO-V3/dataset/config-aug.json \
    -model_file evaluation/checkpoint/MAESTRO-V3/model-with-aug-data_005_006.pkl

python evaluation/transcribe_new_files.py \
    -input_file_to_transcribe ../data/Johann_Sebastian_Bach_Glenn_Gould_-_The_Well-Tempered_Clavier_Book_1_Fugue_No._6_in_D_Minor_BWV_851.mp3 \
    -output_file ../data/test-bach-with-diff-models/bach__aug-hft__006_009.midi \
    -f_config corpus/MAESTRO-V3/dataset/config-aug.json \
    -model_file evaluation/checkpoint/MAESTRO-V3/model-with-aug-data_006_009.pkl

python evaluation/transcribe_new_files.py \
    -input_dir_to_transcribe evaluation/glenn-gould-bach-data \
    -output_dir evaluation/glenn-gould-bach-data/base-model \
    -f_config corpus/MAESTRO-V3/dataset/config-orig.json \
    -model_file evaluation/checkpoint/MAESTRO-V3/model_016_003.pkl
    
python evaluation/transcribe_new_files.py \
    -input_dir_to_transcribe evaluation/glenn-gould-bach-data \
    -output_dir evaluation/glenn-gould-bach-data/aug-model-006-009 \
    -f_config corpus/MAESTRO-V3/dataset/config-aug.json \
    -model_file evaluation/checkpoint/MAESTRO-V3/model-with-aug-data_006_009.pkl    

box 11: 1727 - 2173 running final sweep
box 12: 2173 - 2619 running final sweep
box 13: 2619 - 3065 still running
box 14: 3065 - 3511  running final sweep
box 10 : 3511 - 3957 running final sweep
box 9 : 3957 - 4403 still running
box 7: 4403 - 4849 running final sweep
box 6: 4849 - 5295 running final sweep
box 5: 5295 - 5741 running final sweep
box 4: 5741 - 6185 still running

cd /mnt/data10/spangher/aira-dl/hFT-Transformer
python evaluation/transcribe_new_files.py \
    -input_dir_to_transcribe '../corpus/*' \
    -output_dir ../data/2024-01-09__full-corpus__006-009-model/ \
    -f_config corpus/MAESTRO-V3/dataset/config-aug.json \
    -model_file evaluation/checkpoint/MAESTRO-V3/model-with-aug-data_006_009.pkl \
    -start_index 5741 -end_index 6185
"""
