#! python
import os
import argparse
import pickle
import json
import sys
import glob
sys.path.append(os.getcwd())
from model import amt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir_to_transcribe', help='file list', default='../corpus/MAESTRO-V3/list/test.list')
    parser.add_argument('-d_wav', help='corpus wav directory', default='../corpus/MAESTRO-V3/wav')
    parser.add_argument('-f_config', help='config json file', default='../corpus/config.json')

    parser.add_argument('-model_file', help='input model file', default='best_model.pkl')
    parser.add_argument('-mode', help='mode to transcript (combination|single)', default='combination')

    parser.add_argument('-thred_mpe', help='threshold value for mpe detection', type=float, default=0.5)
    parser.add_argument('-thred_onset', help='threshold value for onset detection', type=float, default=0.5)
    parser.add_argument('-thred_offset', help='threshold value for offset detection', type=float, default=0.5)
    parser.add_argument('-n_stride', help='number of samples for offset', type=int, default=0)
    parser.add_argument('-ablation', help='ablation mode', action='store_true')
    args = parser.parse_args()

    # list file
    a_list = glob.glob(os.path.join(args.input_dir_to_transcribe, '*.wav'))

    # config file
    with open(args.f_config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # AMT class
    AMT = amt.AMT(config, args.model_file, verbose_flag=False)

    for fname in a_list:

        print('[' + os.path.basename(fname) + ']')
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

        # output to MIDI
        output_fname = fname.replace('.wav', '_transcribed.mid')
        AMT.note2midi(a_note_2nd_predict, output_fname)

    print('** done **')
