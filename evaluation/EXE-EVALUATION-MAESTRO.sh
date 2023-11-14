#! /bin/bash
CURRENT_DIR=$(pwd)

# sh evaluation/EXE-EVALUATION-MAESTRO.sh model_016_003.pkl valid

FILE_LIST=$CURRENT_DIR/corpus/MAESTRO-V3/list/$2.list
FILE_CONFIG=$CURRENT_DIR/corpus/MAESTRO-V3/dataset/config.json
#FILE_CONFIG=$CURRENT_DIR/corpus/config.json
DIR_FEATURE=$CURRENT_DIR/corpus/MAESTRO-V3/feature
DIR_REFERENCE=$CURRENT_DIR/corpus/MAESTRO-V3/reference
DIR_CHECKPOINT=$CURRENT_DIR/checkpoint/MAESTRO-V3
DIR_RESULT=$CURRENT_DIR/result/MAESTRO-V3
mkdir -p $DIR_RESULT
MODE=combination
OUTPUT=2nd

# inference
python3 $CURRENT_DIR/evaluation/m_inference.py \
  -f_list $FILE_LIST \
  -f_config $FILE_CONFIG \
  -d_cp $DIR_CHECKPOINT \
  -m $1 \
  -d_fe \
  $DIR_FEATURE \
  -d_mpe $DIR_RESULT \
  -d_note $DIR_RESULT \
  -calc_transcript \
  -mode $MODE &&

# (for half-stride)
# python3 $CURRENT_DIR/evaluation/m_inference.py -f_list $FILE_LIST -f_config $FILE_CONFIG -d_cp $DIR_CHECKPOINT -m $1 -d_fe $DIR_FEATURE -d_mpe $DIR_RESULT -d_note $DIR_RESULT -calc_transcript -mode $MODE  -n_stride 32 &&

# mir_eval
python3 $CURRENT_DIR/evaluation/m_transcription.py \
  -f_list $FILE_LIST \
  -d_ref $DIR_REFERENCE \
  -d_est $DIR_RESULT \
  -d_out $DIR_RESULT \
  -output $OUTPUT &&

python3 $CURRENT_DIR/evaluation/m_transcription.py \
  -f_list $FILE_LIST \
  -d_ref $DIR_REFERENCE \
  -d_est $DIR_RESULT \
  -d_out $DIR_RESULT \
  -output $OUTPUT -velocity &&

python3 $CURRENT_DIR/evaluation/m_mpe.py \
  -f_config $FILE_CONFIG \
  -f_list $FILE_LIST \
  -d_ref $DIR_REFERENCE \
  -d_est $DIR_RESULT \
  -d_out $DIR_RESULT \
  -output $OUTPUT \
  -thred_mpe 0.5








python3 evaluation/m_inference.py \
  -f_list corpus/MAESTRO-V3/list/valid.list \
  -f_config corpus/MAESTRO-V3/dataset/config.json \
  -d_cp checkpoint/MAESTRO-V3 \
  -m model_016_003.pkl \
  -d_fe corpus/MAESTRO-V3/feature \
  -d_mpe result/MAESTRO-V3 \
  -d_note result/MAESTRO-V3 \
  -calc_transcript \
  -mode combination