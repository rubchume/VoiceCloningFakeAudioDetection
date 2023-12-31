import logging
import random
from typing import Iterable, List

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


logging.getLogger().setLevel(logging.INFO)


def audio_file_to_mel_spectogram(audio_file, target_sample_rate, truncation_num_samples, mel_spectogram_calculator):
    pcm_samples, sample_rate = torchaudio.load(audio_file)
    pcm_samples = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(pcm_samples)
    resized_samples = torch.zeros((1, truncation_num_samples))
    offset = target_sample_rate
    pcm_samples_fragment = pcm_samples[0, offset:offset + truncation_num_samples]
    resized_samples[0, :len(pcm_samples_fragment)] = pcm_samples_fragment
    spectogram = mel_spectogram_calculator(resized_samples)
    spectogram = torchaudio.transforms.AmplitudeToDB(top_db=80)(spectogram)
    return spectogram


class AudioDumbDataset(Dataset):
    def __init__(
        self,
        audio_files: Iterable,
        target_sample_rate: int,
        num_samples: int,
    ):
        self.audio_files = list(audio_files)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        
        fft_length = 1024
        num_mel_filterbanks = 128
        self.mel_spectogram_calculator = torchaudio.transforms.MelSpectrogram(
            target_sample_rate,
            n_fft=fft_length,
            n_mels=num_mel_filterbanks
        )

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, index):
        audio_file = self.audio_files[index]
        spectogram = audio_file_to_mel_spectogram(
            audio_file,
            self.target_sample_rate,
            self.num_samples,
            self.mel_spectogram_calculator
        )
        return spectogram
        

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
        spectogram = audio_file_to_mel_spectogram(
            audio_file,
            self.target_sample_rate,
            self.num_samples,
            self.mel_spectogram_calculator
        )
        return spectogram, label
    
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
                
