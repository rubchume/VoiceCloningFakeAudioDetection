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
        
    def _create_model(self):
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
        return self.model.forward(x)
    
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
        logits = self.forward(data).logits
        if torch.any(torch.isnan(logits)):
            logging.info("IS NAN")
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
        
