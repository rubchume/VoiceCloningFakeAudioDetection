source variables.env


python tacotron2/train.py --output_directory=outdir --log_directory=logdir --hparams "training_files=${TRAINING_TRANSCRIPTS},validation_files=${VALIDATION_TRANSCRIPTS}"