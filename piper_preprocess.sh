#!/bin/bash
source bash_utilities.sh

source /anaconda/etc/profile.d/conda.sh
conda activate piperttsenv

DATASET=""
RUN_NAME=""
parse_arguments "dataset" DATASET "run_name" RUN_NAME -- $@

python3 -m piper_train.preprocess \
  --language es \
  --input-dir data/$DATASET/ \
  --output-dir artifacts/$RUN_NAME/ \
  --dataset-format ljspeech \
  --single-speaker \
  --sample-rate 22050