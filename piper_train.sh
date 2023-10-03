#!/bin/bash
source bash_utilities.sh

source /anaconda/etc/profile.d/conda.sh
conda activate piperttsenv

DATASET=""
CHECKPOINT=""
EPOCHS=1
parse_arguments \
    "dataset-preprocessed" DATASET \
    "checkpoint" CHECKPOINT \
    "epochs" EPOCHS \
    -- $@


python3 -m piper_train \
    --dataset-dir "$DATASET" \
    --accelerator 'gpu' \
    --devices 1 \
    --batch-size 32 \
    --validation-split 0.0 \
    --num-test-examples 0 \
    --max_epochs $EPOCHS \
    --resume_from_checkpoint "$CHECKPOINT" \
    --checkpoint-epochs 1 \
    --precision 32