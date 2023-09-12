import argparse
import csv
import inspect
import logging
from num2words import num2words
from pathlib import Path
import re

import mlflow
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.utils.text import cleaners
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from custom_formatter import custom_formatter


def make_command(function):
    parser = argparse.ArgumentParser()
    for parameter_name, parameter in inspect.signature(function).parameters.items():
        parser.add_argument(f"--{parameter_name}", type=parameter.annotation if parameter.annotation != inspect._empty else None)
    
    def wrapper():
        args = parser.parse_args()
        return function(**vars(args))
    
    return wrapper


def get_configuration(audio_data: str, output_path: str):
    dataset_config = BaseDatasetConfig(
        meta_file_train="eva_transcript.txt",
        path=audio_data
    )

    audio_config = VitsAudioConfig(
        sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
    )

    config = VitsConfig(
        audio=audio_config,
        run_name="vits_eva",
        batch_size=4,
        eval_batch_size=4,
        eval_split_size=0.1,
        batch_group_size=5,
        num_loader_workers=1,
        num_eval_loader_workers=1,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1,
        use_phonemes=False,
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=False,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=False,
        test_sentences=[
          "Hola me llamo Eva",
          "Soy la clon de su voz. ¿En qué puedo ayudarte?",
        ],
        characters=CharactersConfig(
            characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnñopqrstuvwxyzáéíóú",
            punctuations="!'(),-.:;? ¡¿",
            pad="<PAD>",
            eos='<EOS>',
            bos='<BOS>',
            blank='<BLNK>',
        )
    )
    
    return config

    
@make_command
def main(audio_dataset):
    logging.info("Start training")
    mlflow.autolog()
    with mlflow.start_run() as run:
        output_path = str(Path().resolve())
        
        logging.info("Define configurations")
        configuration = get_configuration(audio_dataset, output_path)

        logging.info("Create audio processor")
        audio_processor = AudioProcessor.init_from_config(configuration)
        
        logging.info("Create tokenizer")
        tokenizer, config = TTSTokenizer.init_from_config(configuration)

        logging.info("Create model")
        model = Vits(configuration, audio_processor, tokenizer, speaker_manager=None)
        
        logging.info("Load samples")
        train_samples, eval_samples = load_tts_samples(
            configuration.datasets[0],
            formatter=custom_formatter,
            eval_split=True,
            eval_split_max_size=config.eval_split_max_size,
            eval_split_size=config.eval_split_size,
        )
        
        logging.info("Create trainer")
        trainer = Trainer(
            TrainerArgs(),
            config,
            output_path,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
        )

        logging.info("Train")
        trainer.fit()
        
        logging.info("Log artifacts")
        mlflow.log_artifacts(trainer.output_path)

    logging.info("Run was finished")
    
    
if __name__ == "__main__":
    main()
