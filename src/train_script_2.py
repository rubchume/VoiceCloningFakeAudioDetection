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


cleaners.spanish_cleaners = spanish_cleaners

    
output_path = str(Path().resolve())
dataset_config = BaseDatasetConfig(
    meta_file_train="eva_transcript.txt",
    path="data",
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

audio_processor = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    formatter=custom_formatter,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

model = Vits(config, audio_processor, tokenizer, speaker_manager=None)

trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()
