import argparse
import inspect
import logging
from pathlib import Path

from azureml.core import Run, Workspace
import mlflow
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from audio_binary_datamodule import DataModule
from cloned_audio_detector import ClonedAudioDetector


logging.getLogger().setLevel(logging.INFO)


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

    try:
        run = Run.get_context()
        experiment = getattr(run, "experiment")
        experiment_name = experiment.name
    except AttributeError:
        experiment_name = "LocalExperiment"
    ws = Workspace.from_config()
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    mlflow.set_experiment(experiment_name)

    logging.info(f"Start experiment {experiment_name}")
    # mlflow.autolog()
    with mlflow.start_run() as run:        
        mlogger = MLFlowLogger(experiment_name=experiment_name, run_id=run.info.run_id)
        detector = ClonedAudioDetector()
        trainer = pl.Trainer(
            logger=mlogger,
            max_epochs=int(max_epochs),
            accelerator="auto",
            log_every_n_steps=10,
            callbacks=[],
            # limit_train_batches=5,
            # limit_val_batches=5,
        )
    
        trainer.fit(detector, data_module)
        trainer.test(detector, data_module)
        trainer.save_checkpoint(checkpoint_path)
    
    logging.info("Finished training")
    

if __name__=="__main__":
    main()
