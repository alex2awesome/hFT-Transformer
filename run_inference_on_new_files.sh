python evaluation/transcribe_new_files.py \
    -input_dir_to_transcribe ../data/test-album \
    -output_dir ../data/test-album-transcribed \
    -f_config corpus/MAESTRO-V3/dataset/config-orig.json \
    -model_file evaluation/checkpoint/MAESTRO-V3/model_016_003.pkl

python evaluation/transcribe_new_files.py \
    -input_dir_to_transcribe ../data/test-album \
    -output_dir ../data/test-album-transcribed \
    -f_config corpus/MAESTRO-V3/dataset/config-aug.json \
    -model_file evaluation/checkpoint/MAESTRO-V3/model-with-aug-data_005_006.pkl