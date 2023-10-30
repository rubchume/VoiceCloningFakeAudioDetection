from enum import Enum
from collections import defaultdict
import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from transformers import AutoModelForAudioClassification


class Stage(Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


class ClonedAudioDetector(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._create_model()
        self._prepare_metrics()
        
    def _create_model_old(self):
        num_labels = 2

        label2id = dict(
            cloned=1,
            real=0)

        id2label = {
            1: "cloned",
            0: "real"
        }

        self.model = AutoModelForAudioClassification.from_pretrained(
            "facebook/wav2vec2-base",
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label
        )
        
    def _create_model(self):
        """https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5"""
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        nn.init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        nn.init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
             
    def _prepare_metrics(self):
        self.precision = torchmetrics.Precision(task='binary')
        self.recall = torchmetrics.Recall(task='binary')
        self.f1 = torchmetrics.F1Score(task='binary')
        self.confmat = torchmetrics.ConfusionMatrix(task="binary")

        self.targets_scores = {}
        self.targets_predicted = {}
        self.targets = {}
        
        self._reset_target_registries(Stage.TRAIN)
        self._reset_target_registries(Stage.VALIDATION)
        self._reset_target_registries(Stage.TEST)
 
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x
    
    def criterion(self, logits, labels):
        return nn.functional.cross_entropy(logits, labels)
    
    def training_step(self, batch, batch_index):
        return self._step(batch, Stage.TRAIN)

    def validation_step(self, batch, batch_index):
        return self._step(batch, Stage.VALIDATION)
        
    def test_step(self, batch, batch_index):
        return self._step(batch, Stage.TEST)
    
    def on_train_epoch_start(self):
        self._reset_target_registries(Stage.TRAIN)
    
    def on_train_epoch_end(self):
        self._log_epoch_metrics(Stage.TRAIN)
    
    def on_validation_epoch_start(self):
        self._reset_target_registries(Stage.VALIDATION)
    
    def on_validation_epoch_end(self):
        self._log_epoch_metrics(Stage.VALIDATION)
    
    def on_test_epoch_start(self):
        self._reset_target_registries(Stage.TEST)
    
    def on_test_epoch_end(self):
        self._log_epoch_metrics(Stage.TEST)
                                
    def _reset_target_registries(self, stage: Stage):
        self.targets_scores[stage] = []
        self.targets_predicted[stage] = []
        self.targets[stage] = []

    def _step(self, batch, stage: Stage):
        audios, targets = batch
        logits, targets_predicted = self._predict(audios)
        self.targets_scores[stage].append(logits)
        self.targets_predicted[stage].append(targets_predicted)
        self.targets[stage].append(targets)
        
        loss = self.criterion(logits, targets)
        
        metric_name = {
            stage.TRAIN: "train_loss",
            stage.VALIDATION: "val_loss",
            stage.TEST: "test_loss",
        }
        
        self.log(metric_name[stage], loss, prog_bar=True)
        return loss
        
    def _predict(self, data):
        logits = self.forward(data)
        targets_predicted = (logits[:, 1] > logits[:, 0]) * 1
        return logits, targets_predicted
        
    def _log_epoch_metrics(self, stage: Stage):
        targets_predicted = torch.cat(self.targets_predicted[stage], dim=0).squeeze()
        targets = torch.cat(self.targets[stage], dim=0)

        precision = self.precision(targets_predicted, targets)
        recall = self.recall(targets_predicted, targets)
        f1_score = self.f1(targets_predicted, targets)

        self.log(f'{stage.value}_precision', precision, prog_bar=True)
        self.log(f'{stage.value}_recall', recall, prog_bar=True)
        self.log(f'{stage.value}_f1', f1_score, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def get_targets(self, stage: Stage):
        return torch.cat(self.targets[stage], dim=0).to(torch.device("cpu"))
    
    def get_targets_scores(self, stage: Stage):
        return torch.cat(self.targets_scores[stage], dim=0).squeeze().to(torch.device("cpu"))
    
    def get_targets_predicted(self, stage: Stage):
        return torch.cat(self.targets_predicted[stage], 0).squeeze().to(torch.device("cpu"))
        
