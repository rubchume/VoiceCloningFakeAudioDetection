import logging
import random
from typing import Iterable, List

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


logging.getLogger().setLevel(logging.INFO)


class AudioBinaryDataset(Dataset):
    def __init__(
        self,
        negative_audio_files: Iterable,
        postive_audio_files: Iterable,
        target_sample_rate: int,
        num_samples: int,
        max_imbalance=1,
        random_seed=0,
    ):
        self.negative_audio_files = list(negative_audio_files)
        self.positive_audio_files = list(postive_audio_files)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        
        self.random_instance = random.Random(random_seed)
        
        negative_samples, positive_samples = self._undersample_unbalanced_dataset(
            self.negative_audio_files,
            self.positive_audio_files,
            max_imbalance
        )
        
        negative_samples_with_label = [
            (sample, 0)
            for sample in negative_samples
        ]
        
        positive_samples_with_label = [
            (sample, 1)
            for sample in positive_samples
        ]
        
        self.samples = self.random_instance.sample(
            negative_samples_with_label + positive_samples_with_label,
            len(negative_samples_with_label) + len(positive_samples_with_label)
        )
        
        fft_length = 1024
        num_mel_filterbanks = 128
        self.mel_spectogram_calculator = torchaudio.transforms.MelSpectrogram(
            target_sample_rate,
            n_fft=fft_length,
            n_mels=num_mel_filterbanks
        )
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        audio_file, label = self.samples[index]
        # from pathlib import Path; logging.info(f"This file (audio_file) exist? {Path(audio_file).is_file()}")
        pcm_samples, sample_rate = torchaudio.load(audio_file)
        # raise RuntimeError("Borrar")
        pcm_samples = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)(pcm_samples)
        resized_samples = torch.zeros((1, self.num_samples))
        resized_samples[0, :pcm_samples.shape[1]] = pcm_samples[0, :self.num_samples]
        resized_samples /= resized_samples.max()
        spectogram = self._get_mel_spectogram(resized_samples)
        return spectogram, label
    
    def _get_mel_spectogram(self, pcm_samples):
        spectogram = self.mel_spectogram_calculator(pcm_samples)
        return torchaudio.transforms.AmplitudeToDB(top_db=80)(spectogram)
    
    def _undersample_unbalanced_dataset(self, dataset_A: List, dataset_B: List, max_imbalance):
        if len(dataset_A) > len(dataset_B):
            dataset_big = dataset_A
            dataset_small = dataset_B
            a_bigger_than_b = True
        else:
            dataset_big = dataset_B
            dataset_small = dataset_A
            a_bigger_than_b = False
        
        if max_imbalance < 1:
            max_imbalance = 1 / max_imbalance
            
        max_samples = int(len(dataset_small) * max_imbalance)
        samples_big = self.random_instance.sample(dataset_big, min(max_samples, len(dataset_big)))
        samples_small = self.random_instance.sample(dataset_small, len(dataset_small))
        
        if a_bigger_than_b:
            return samples_big, samples_small
        else:
            return samples_small, samples_big
                
