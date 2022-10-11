"""General trainer class that uses huggingface accelerate, and general metric
class
"""

import accelerate
from accelerate import logging
import collections
import contextlib
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR

from transformers import (get_linear_schedule_with_warmup, 
                          get_cosine_schedule_with_warmup)

class Metric:
    """General Metric class"""

    @property
    def score(self) -> float:
        """Main metric score used for comparison"""
        raise NotImplementedError

class Trainer:
    """General trainer class that uses huggingface """

    def __init__(self,
                accelerator: accelerate.Accelerator,
                logger: logging.MultiProcessAdapter,
                model: nn.Module,
                train_dataloader: DataLoader,
                dev_dataloader: DataLoader,
                optimizer: Optimizer,
                scheduler_type: str = "no",
                gamma: float = None,
                warmup_ratio: float = None,
                warmup_steps: int = None,
                max_epochs: int = 1,
                max_grad_norm: float = None,
                patience: int = 1,
                log_batch_frequency: int = 1,
                evaluate_train: bool = False,
                save_model: bool = False,
                save_plot: bool = False,
                save_tensors: bool = False,
                save_tensors_names: list[str] = None,
                save_dir: str = None,
                verbose: bool = False
                ) -> None:
        """Initializer for the general trainer class that uses accelerate to
        train your model.

        Args:
            accelerator: Instance of the Accelerator class.
            logger: Instance of the Accelerator logger used for distributed
                logging.
            model: Torch nn.Module subclass to train.
            train_dataloader: Train set dataloader.
            dev_dataloader: Dev set dataloader.
            optimizer: Optimizer.
            scheduler_type: Type of scheduler to use, can be "linear", "cosine",
                "exponential", or "no". "no" means no scheduler is used and
                learning rate is kept constant.
            gamma: Multiplier to be used in exponential scheduler.
            warmup_ratio: Ratio of training steps to use in the scheduler's
                warmup. warmup_steps has to be None if you want to use this
                parameter.
            warmup_steps: Number of steps to use in the scheduler's warmup.
                This parameter supercedes warmup_ratio.
            max_epochs: Maximum number of epochs to train for.
            max_grad_norm: Maximum norm of gradient to be used in gradient
                clipping. If None, gradient clipping is not done.
            patience: Maximum number of epochs to wait for development set
                performance to improve before early-stopping.
            log_batch_frequency: If verbose is true, training loop logs training
                loss and timing information after every log_batch_frequency
                batches.
            evaluate_train: Whether to evaluate on the training set.
            save_model: Whether to save model after every epoch.
            save_plot: Whether to create and save matplotlib plot of training
                and development loss.
            save_tensors: Whether to save the tensors and logits.
            save_tensors_names: List of tensor names which are to be saved. If
                none, all tensors are saved.
            save_dir: Directory to which the model weights, ground truth, and
                predictions will be saved.
            verbose: If true, log training loss after every log_batch_frequency
                batches, and timer messages.
        """
        self.accelerator = accelerator
        self.logger = logger
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.optimizer = optimizer
        self.scheduler = None
        self.scheduler_type = scheduler_type
        self.gamma = gamma
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.log_batch_frequency = log_batch_frequency
        self.evaluate_train = evaluate_train
        self.save_model = save_model
        self.save_plot = save_plot
        self.save_tensors = save_tensors
        self.save_tensors_names = save_tensors_names
        self.save_dir = save_dir
        self.verbose = verbose

        assert self.scheduler_type in ["linear", "cosine", "exponential", "no"], (
                "scheduler_type should be linear, cosine, exponential, or no")
        if self.scheduler_type != "no":
            if self.scheduler_type in ["linear", "cosine"]:
                assert (self.warmup_ratio is not None or 
                        self.warmup_steps is not None), (
                        "Set warmup_ratio or warmup_steps "
                        "if you are using linear/cosine scheduler")
            if self.scheduler_type == "exponential":
                assert self.gamma is not None, (
                    "Set gamma if you are using exponential scheduler")
        
        if self.save_model or self.save_plot or self.save_tensors:
            assert self.save_dir is not None, (
                "Set save_dir if you are saving model, plots, or tensors")
        
        self.n_training_samples = len(self.train_dataloader.dataset)
        self.n_dev_samples = len(self.dev_dataloader.dataset)
        self.model.eval()
        self.model.device = self.accelerator.device
    
    def _log(self, message):
        """Log in distributed setting"""
        self.logger.info(message)
    
    @contextlib.contextmanager
    def _timer(self, message):
        """Context manager for timing a codeblock"""
        start_time = time.time()
        if self.verbose:
            self._log(f"{message} [start timer]")
        yield
        time_taken = time.time() - start_time
        time_taken_str = self._convert_float_seconds_to_time_string(time_taken)
        if self.verbose:
            self._log(f"{message} [end timer: time taken = {time_taken_str}]")

    def _convert_float_seconds_to_time_string(self, seconds: float) -> str:
        """Convert seconds to h m s format"""
        seconds = int(seconds)
        minutes, seconds = seconds//60, seconds%60
        hours, minutes = minutes//60, minutes%60
        return f"{hours}h {minutes}m {seconds}s"
    
    def _save_model(self, model: nn.Module, directory: str):
        """Save model's weights to directory with filename `model.pt`.

        Args:
            model: Torch nn.Module.
            directory: Directory where model's weights will be saved.
        """
        self.accelerator.save(
            model.state_dict(), os.path.join(directory, "model.pt"))

    def _save_tensors(self, directory: str, **tensors):
        """Save the tensors to directory.

        Args:
            directory: Directory where the tensors will be saved.
            tensors: Dictionary of tensors.
        """
        for name, pt in tensors.items():
            if self.save_tensors_names is None or (
                name in self.save_tensors_names):
                self.accelerator.save(pt, os.path.join(directory, f"{name}.pt"))
    
    def _training_step(self, batch: dict[str, torch.Tensor]) -> float:
        """Trains the model for a single batch.

        Args:
            batch: Dictionary of tensor name to tensor
        
        Returns:
            Loss value
        """
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            with self.accelerator.autocast():
                batch_loss, _ = self.model(**batch)
            self.accelerator.backward(batch_loss)
            if self.optimizer.gradient_state.sync_gradients and (
               self.max_grad_norm is not None):
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.scheduler_type != "no" and (
               not self.accelerator.optimizer_step_was_skipped):
                self.scheduler.step()
        return batch_loss.detach().item()

    def _log_train(self):
        """Log loss and time during training"""
        average_train_loss = np.mean(self.running_train_losses)
        average_train_time = np.mean(self.running_train_times)
        time_remaining = average_train_time * (
            self.n_train_batches - self.batch_index - 1)
        time_remaining_str = self._convert_float_seconds_to_time_string(
            time_remaining)
        average_train_time_str = self._convert_float_seconds_to_time_string(
            average_train_time)
        prefix = f"Epoch {self.epoch + 1}: Batch {self.batch_index + 1}"
        self._log(f"{prefix} Average training loss @ batch = "
                  f"{average_train_loss:.4f}")
        self._log(f"{prefix} Average training time taken @ batch = "
                  f"{average_train_time_str}")
        self._log(f"{prefix} Estimated training time remaining for epoch "
                    f"{self.epoch + 1} = {time_remaining_str}")

    def evaluate(self, **tensors) -> Metric:
        """Evaluate the output of inference"""
        raise NotImplementedError

    def _infer(self, dataloader: DataLoader, model: nn.Module) -> (
        tuple[dict[str, torch.Tensor], float]):
        """Run inference on the dataloader.
        Args:
            dataloader: PyTorch dataloader.
            model: PyTorch module.
        
        Returns:
            Tensors and Loss.
        """
        # Initialize variables
        model.eval()
        tensors: dict[str, list[torch.Tensor]] = collections.defaultdict(list)
        n_batches = len(dataloader)
        batch_losses = []

        # Inference Loop
        with torch.no_grad():
            running_batch_times = []
            for i, batch in enumerate(dataloader):
                start_time = time.time()
                batch_loss, batch_logits = model(**batch)
                batch_losses.append(batch_loss.item())
                batch_logits = self.accelerator.gather_for_metrics(batch_logits)
                batch = self.accelerator.gather_for_metrics(batch)
                batch["logits"] = batch_logits
                for name, tensor in batch.items():
                    tensors[name].append(tensor)
                time_taken = time.time() - start_time
                running_batch_times.append(time_taken)

                # Log after log_batch_frequency batches
                if self.verbose and (i + 1) % self.log_batch_frequency == 0:
                    average_time_per_batch = np.mean(running_batch_times)
                    estimated_time_remaining = (n_batches - i - 1) * (
                                                average_time_per_batch)
                    average_time_per_batch_str = (
                        self._convert_float_seconds_to_time_string(
                            average_time_per_batch))
                    estimated_time_remaining_str = (
                        self._convert_float_seconds_to_time_string(
                            estimated_time_remaining))
                    running_batch_times = []
                    prefix = f"Epoch {self.epoch + 1}: Batch {i + 1}"
                    self._log(f"{prefix} Average inference time @ batch = "
                                f"{average_time_per_batch_str}")
                    self._log(f"{prefix} Estimated inference time remaining = "
                                f"{estimated_time_remaining_str}")

        # Average batch losses
        average_loss = np.mean(batch_losses)
        self._log(f"Epoch {self.epoch + 1}: Average inference loss = "
                    f"{average_loss:.4f}")

        # Concat tensors
        output: dict[str, torch.Tensor] = {}
        for name, tensor_list in tensors.items():
            output[name] = torch.cat(tensor_list, dim=0)
        return output, average_loss
    
    def _infer_and_evaluate(self, dataloader: DataLoader, 
                            train: bool = False) -> tuple[Metric, float]:
        """Run inference and evaluation on dataloader, and save tensors.

        Args:
            dataloader: Torch dataloader
            train: Whether the dataloader is of the train set

        Returns:
            Evaluation metric and loss.
        """
        name = "Train" if train else "Dev"
        inference_output, loss = self._infer(dataloader, self.model)
        metric = self.evaluate(**inference_output)
        self._log(f"Epoch {self.epoch + 1}: {name} Performance = {metric}")
        self.accelerator.wait_for_everyone()
        if self.save_tensors:
            save_dir = self.epoch_train_dir if train else self.epoch_dev_dir
            self._log(f"Epoch {self.epoch + 1}: Save {name} Tensors")
            self._save_tensors(save_dir, **inference_output)
        return metric, loss

    def _plot_loss_graph(self):
        """Plot loss graph"""
        filename = os.path.join(self.save_dir, "loss.png")
        plt.close("all")
        plt.plot(np.arange(self.epoch + 1), self.epoch_train_losses,
                 label="train_loss", lw=4)
        plt.plot(np.arange(self.epoch + 1), self.epoch_dev_losses,
                 label="dev_loss", lw=4)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curve")
        plt.savefig(filename)

    def _early_stopping(self, dev_metric: Metric) -> bool:
        """Early stopping code. Return true if training needs to stop"""
        self._log(f"Epoch {self.epoch + 1}: Checking for early-stopping")
        dev_score = dev_metric.score
        if self.best_dev_score is None or dev_score > self.best_dev_score:
            self.epochs_left = self.patience
            self.best_epoch = self.epoch + 1
            if self.best_dev_score is not None:
                delta = 100 * (dev_score - self.best_dev_score)
                self._log(f"Epoch {self.epoch + 1}: "
                          f"Dev score improved by {delta:.1f}")
            self.best_dev_score = dev_score
        else:
            self.epochs_left -= 1
            delta = 100 * (self.best_dev_score - dev_score)
            self._log(f"Epoch {self.epoch + 1}: "
                      f"Dev score is {delta:.1f} lower than best Dev score "
                      f"({100*self.best_dev_score:.1f})")
            self._log(f"Epoch {self.epoch + 1}: "
                f"{self.epochs_left} epochs left until Dev score to improve to"
                " avoid early-stopping!")
        if self.epochs_left == 0:
            self._log(f"Epoch {self.epoch + 1}: Early stopping!")
            return True
        return False

    def run(self):
        """Run training"""
        self._log("")

        # Early stopping variables
        self.best_dev_score = None
        self.best_epoch = None
        self.epochs_left = self.patience
        save = self.save_model or self.save_tensors
        
        # Accelerate model, dataloaders, and optimizer
        (self.model, self.train_dataloader, self.dev_dataloader,
         self.optimizer) = (self.accelerator.prepare(
            self.model, self.train_dataloader, self.dev_dataloader, 
            self.optimizer))

        # Log number of training and inference batches, 
        # and number of training steps
        self.n_train_batches = len(self.train_dataloader)
        self.n_dev_batches = len(self.dev_dataloader)
        effective_train_batch_size = round(
            self.n_training_samples/self.n_train_batches)
        effective_dev_batch_size = round(self.n_dev_samples/self.n_dev_batches)
        n_training_steps = self.max_epochs * self.n_train_batches
        self._log("Effective train batch size = "
                        f"{effective_train_batch_size}")
        self._log("Effective dev batch size = "
                        f"{effective_dev_batch_size}")
        self._log(f"Number of training batches = {self.n_train_batches}")
        self._log(f"Number of inference batches = {self.n_dev_batches}")
        self._log(f"Number of training steps = {n_training_steps}")
        self._log("")

        # Initialize and accelerate scheduler
        if self.scheduler_type != "no":
            if self.scheduler_type in ["linear", "cosine"]:
                n_warmup_steps = self.warmup_steps if (
                    self.warmup_steps is not None) else (
                        int(self.warmup_ratio * n_training_steps))
                if self.scheduler_type == "linear":
                    self.scheduler = get_linear_schedule_with_warmup(
                        self.optimizer, num_warmup_steps=n_warmup_steps,
                        num_training_steps=n_training_steps)
                else:
                    self.scheduler = get_cosine_schedule_with_warmup(
                        self.optimizer, num_warmup_steps=n_warmup_steps,
                        num_training_steps=n_training_steps)
                self._log(f"Number of warmup steps = {n_warmup_steps}")
            else:
                self.scheduler = ExponentialLR(self.optimizer, gamma=self.gamma)
            self.scheduler = self.accelerator.prepare_scheduler(self.scheduler)
        
        # Plot variables
        self.epoch_train_losses = []
        self.epoch_dev_losses = []
        
        # Training and evaluation loop
        for self.epoch in range(self.max_epochs):
            
            # Create epoch directories
            if save:
                epoch_dir = os.path.join(self.save_dir, 
                                         f"epoch_{self.epoch + 1}")
                self.epoch_train_dir = os.path.join(epoch_dir, "train")
                self.epoch_dev_dir = os.path.join(epoch_dir, "dev")
                os.makedirs(self.epoch_train_dir, exist_ok=True)
                os.makedirs(self.epoch_dev_dir, exist_ok=True)

            # Training examine variables
            self.model.train()
            self.train_losses = []
            self.running_train_losses = []
            self.running_train_times = []

            # Training for one epoch
            with self._timer(f"Epoch {self.epoch + 1}: Training"):
                for self.batch_index, batch in enumerate(self.train_dataloader):
                    batch_start_time = time.time()
                    batch_loss = self._training_step(batch)
                    batch_time_taken = time.time() - batch_start_time
                    self.train_losses.append(batch_loss)
                    self.running_train_losses.append(batch_loss)
                    self.running_train_times.append(batch_time_taken)
                    if (self.batch_index + 1) % self.log_batch_frequency == 0:
                        if self.verbose:
                            self._log_train()
                        self.running_train_losses = []
                        self.running_train_times = []
            average_train_loss = np.mean(self.train_losses)
            self._log(f"Epoch {self.epoch + 1}: Average training loss = "
                        f"{average_train_loss:.4f}")
            self.epoch_train_losses.append(average_train_loss)

            # Wait for all process to complete
            self.accelerator.wait_for_everyone()

            # Save model
            if self.save_model:
                self._log(f"Epoch {self.epoch + 1}: Save model")
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                self._save_model(unwrapped_model, epoch_dir)

            # Inference and evaluation on training set
            if self.evaluate_train:
                with self._timer(f"Epoch {self.epoch + 1}: Inference on Train"):
                    self._infer_and_evaluate(self.train_dataloader, train=True)
        
            # Inference and evaluation on dev set
            with self._timer(f"Epoch {self.epoch + 1}: Inference on Dev"):
                dev_metric, dev_loss = self._infer_and_evaluate(
                    self.dev_dataloader)
            self.epoch_dev_losses.append(dev_loss)

            # Plot loss graphs
            if self.save_plot:
                self._plot_loss_graph()

            # Early-stopping
            if self._early_stopping(dev_metric):
                break
            self._log(f"Epoch {self.epoch + 1}: Done")
            self._log("")

        self._log(f"Best Dev score = {100*self.best_dev_score:.1f}")
        self._log(f"Best epoch = {self.best_epoch}")