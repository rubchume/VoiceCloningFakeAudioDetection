import random
from typing import Iterable, List

import numpy as np
from pydub import AudioSegment
import torch
from torch.utils.data import Dataset


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
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        audio_file, label = self.samples[index]
        audio_segment = AudioSegment.from_file(audio_file)
        audio_resampled = audio_segment.set_frame_rate(self.target_sample_rate)
        pcm_samples = self._bytes_to_numpy(
            audio_resampled.raw_data,
            audio_resampled.sample_width
        )
        resized_samples = np.zeros(self.num_samples)
        resized_samples[:len(pcm_samples)] = pcm_samples[:self.num_samples]
        return torch.Tensor(resized_samples), label
    
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
    
    @staticmethod
    def _bytes_to_numpy(bytes_stream: bytes, sample_width=2) -> np.array:
        """
        sample_width: number of bytes per sample
        """
        dtype_map = {
            1: np.int8,
            2: np.int16,
            4: np.int32
        }

        if sample_width not in dtype_map:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        return np.frombuffer(bytes_stream, dtype=dtype_map[sample_width])
