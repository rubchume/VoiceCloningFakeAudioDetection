import argparse
import inspect
import logging
from pathlib import Path

import os
import sys

logging.getLogger().setLevel(logging.INFO)
logging.info("I am here")
logging.info(sys.executable)


import pkg_resources

def list_installed_packages():
    installed_packages = [(d.project_name, d.version) for d in pkg_resources.working_set]
    return installed_packages

for name, version in list_installed_packages():
    logging.info(f"{name}=={version}")


import mlflow
import pandas as pd
import pytorch_lightning as pl

from audio_binary_datamodule import DataModule
from cloned_audio_detector import ClonedAudioDetector


logging.basicConfig(level=logging.INFO)


def make_command(function):
    parser = argparse.ArgumentParser()
    for parameter_name, parameter in inspect.signature(function).parameters.items():
        parser.add_argument(f"--{parameter_name}", type=parameter.annotation if parameter.annotation != inspect._empty else None)
    
    def wrapper():
        args = parser.parse_args()
        return function(**vars(args))
    
    return wrapper


@make_command
def main(
    real_voices_dataset,
    cloned_voices_dataset,
    real_audio_files,
    cloned_audio_files,
    checkpoint_path,
    max_epochs=3
):
    logging.info("Start training")
    
    real_audio_files_list = Path(real_audio_files).read_text().splitlines()
    cloned_audio_files_list = Path(cloned_audio_files).read_text().splitlines()

    real_audio_files_list = [real_voices_dataset + audio_file for audio_file in real_audio_files_list]
    cloned_audio_files_list = [cloned_voices_dataset + audio_file for audio_file in cloned_audio_files_list]

    data_module = DataModule(4, 16000, 64000, real_audio_files_list, cloned_audio_files_list)

    logging.info("Start experiment")
    mlflow.autolog()
    with mlflow.start_run() as run:        
        detector = ClonedAudioDetector()
        trainer = pl.Trainer(
            max_epochs=int(max_epochs),
            accelerator="auto",
            log_every_n_steps=10,
            callbacks=[],
            limit_train_batches=5,
            limit_val_batches=5,
        )

    
        trainer.fit(detector, data_module)
        trainer.test(detector, data_module)
        trainer.save_checkpoint(checkpoint_path)
    
    logging.info("Finished training")
    

if __name__=="__main__":
    main()
