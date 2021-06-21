import logging
from ..utils.utils import LogProgress, bold
import time
import os
import json
import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets
import torchvision.models as models
from torchvision.transforms import ToTensor

logger = logging.getLogger(__name__)

class MetricTracker:
    def __init__(self):
        self.tracker = {}
        self.tracker["train_metrics"] = []
        self.tracker["val_metrics"] = []
        self.tracker["train_loss"] = []
        self.tracker["val_loss"] = []

    def update(self, key, metrics, loss):
        if key == "train":
            self.tracker["train_metrics"].append(metrics)
            self.tracker["train_loss"].append(loss)
        elif key == "val":
            self.tracker["val_metrics"].append(metrics)
            self.tracker["val_loss"].append(loss)
        else:
            logger.error(f"The key: '{key}' is not available. Available: 'train' and 'val'.")


    def get_result(self):
        return self.tracker


class Trainer(object):
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 metrics,
                 device,
                 data_loader,
                 valid_data_loader,
                 args, 
                 metric_tracker=None,
                 lr_scheduler=None,
                 gradient_clippers=None):
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.model.to(self.device)
        self.metric_tracker = MetricTracker()
        self.num_prints = 1
        self.history_file = args.history_file
        self.history = []
        self.epochs = args.epochs
        self.args = args
        
    def _train_one_epoch(self, epoch):
        self.model.train()

        running_loss = 0
        running_metrics = 0
        size = 0

        name = f"Train | Epoch {epoch + 1}"
        logprog = LogProgress(logger, 
                              self.data_loader,
                              updates=self.num_prints,
                              name=name)
        
        for batch_idx, (data, target) in enumerate(logprog):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            metrics = self.metrics(output, target)

            size += len(target)
            running_loss += loss.item()
            running_metrics += metrics
            
            logprog.update(loss=format(running_loss / (batch_idx + 1), ".5f"))

        output = {"loss": running_loss / size,
                  "metrics": running_metrics / size}
        del size, running_loss, running_metrics
        return output
            
    def _eval_one_epoch(self, epoch):
        self.model.eval()
        running_loss = 0
        running_metrics = 0
        size = 0
        
        name = f"Valid | Epoch {epoch + 1}"
        logprog = LogProgress(logger, 
                              self.valid_data_loader,
                              updates=self.num_prints,
                              name=name)
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(logprog):
                input, target = input.to(self.device), target.to(self.device)
                
                output = self.model.forward(input)
                loss = self.criterion(output, target)
                metrics = self.metrics(output, target)

                size += len(target)
                running_loss += loss.item()
                running_metrics += metrics
                
                logprog.update(loss=format(running_loss / (batch_idx + 1), ".5f"))

        output = {"loss": running_loss / size,
                  "metrics": running_metrics / size}
        del size, running_loss, running_metrics
        return output

    def train(self):
        for epoch in range(len(self.history), self.epochs):
            if self.history:
                logger.info("Replaying metrics from previous run")
            for epoch, metrics in enumerate(self.history):
                info = " ".join(f"{k}={v:.5f}" for k, v in metrics.items())
                logger.info(f"Epoch {epoch}: {info}")
                
            logger.info('-' * 70)
            logger.info("Training...")
            
            start = time.time()
            train_output = self._train_one_epoch(epoch)
            self.metric_tracker.update("train",
                                       train_output["metrics"],
                                       train_output["loss"])
            logger.info(bold(f'Train Summary | End of Epoch {epoch + 1} | '
                             f'Time {time.time() - start:.2f}s | Train Loss {train_output["loss"]:.5f}'))
            
            logger.info('-' * 70)
            valid_output = self._eval_one_epoch(epoch)
            self.metric_tracker.update("val",
                        valid_output["metrics"],
                        valid_output["loss"])
            logger.info(bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                             f'Time {time.time() - start:.2f}s | Train Loss {valid_output["loss"]:.5f}'))
            
            metrics = {"metrics": valid_output["metrics"],
                       "loss": valid_output["loss"]}
            self.history.append(metrics)
            
            info = " | ".join(
                f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
            logger.info('-' * 70)
            logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))
            
            json.dump(self.history, open(self.history_file, "w"), indent=2)
                # Save model each epoch
            if self.checkpoint:
                self._serialize(self.checkpoint)
                logger.debug("Checkpoint saved to %s",
                            self.checkpoint.resolve())
        
        return self.metric_tracker.get_result()

    def _save_checkpoint(self, epoch, filepath, save_best=False):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _serialize(self, path):
        package = {}
        package['model'] = serialize_model(self.model)
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        torch.save(package, path)
            
    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
