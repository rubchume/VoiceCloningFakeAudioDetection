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


def make_command(function):
    parser = argparse.ArgumentParser()
    for parameter_name, parameter in inspect.signature(function).parameters.items():
        parser.add_argument(f"--{parameter_name}", type=parameter.annotation if parameter.annotation != inspect._empty else None)
    
    def wrapper():
        args = parser.parse_args()
        return function(**vars(args))
    
    return wrapper


def custom_formatter(root_path, meta_file, **kwargs):
    speaker_name = "Eva"
    with (Path(root_path) / meta_file).open("r", encoding="utf-8") as file:
        return [
            {"text": text, "audio_file": str(Path(root_path) / file_name), "speaker_name": speaker_name, "root_path": root_path}
            for file_name, text in csv.reader(file, delimiter="|")
        ]


def spanish_cleaners(text):
    """Pipeline for Spanish text"""
    text = text.lower()
    text = numbers_to_words(text)
    text = remove_unknown_characters(text)
    return text


def numbers_to_words(text):
    def create_replacement(match):
        number = match.group(0)
        return num2words(number, lang="es")

    return re.sub(r"\d+", create_replacement, text)
    
    
def remove_unknown_characters(text):
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnñopqrstuvwxyzáéíóú"
    punctuations = "!'(),-.:;? ¡¿"
    
    return re.sub(fr"[^{characters + punctuations}]", "", text)


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
        text_cleaner="spanish_cleaners",
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
        
    
cleaners.spanish_cleaners = spanish_cleaners

    
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
