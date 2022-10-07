"""General trainer class that uses huggingface accelerate, and general metric
class
"""

import accelerate
from accelerate import logging
import collections
import contextlib
import numpy as np
import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from transformers import get_linear_schedule_with_warmup

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
                use_scheduler: bool = False,
                warmup_ratio: float = None,
                warmup_steps: int = None,
                max_epochs: int = 1,
                max_grad_norm: float = None,
                patience: int = 1,
                log_batch_frequency: int = 1,
                evaluate_train: bool = False,
                save_model: bool = False,
                save_tensors: bool = False,
                save_tensors_names: list[str] = None,
                save_dir: str = None
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
            use_scheduler: Whether to use scheduler.
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
            log_batch_frequency: Training loop logs training loss and timing
                information after every log_batch_frequency batches.
            evaluate_train: Whether to evaluate on the training set.
            save_model: Whether to save model after every epoch.
            save_tensors: Whether to save the tensors of development set, along
                with the logits.
            save_tensors_names: List of tensor names which are to be saved. If
                none, all tensors are saved.
            save_dir: Directory to which the model weights, ground truth, and
                predictions will be saved.
        """
        self.accelerator = accelerator
        self.logger = logger
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.optimizer = optimizer
        self.scheduler = None
        self.use_scheduler = use_scheduler
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.log_batch_frequency = log_batch_frequency
        self.evaluate_train = evaluate_train
        self.save_model = save_model
        self.save_tensors = save_tensors
        self.save_tensors_names = save_tensors_names
        self.save_dir = save_dir

        if self.use_scheduler:
            assert (self.warmup_ratio is not None or 
                    self.warmup_steps is not None), (
                    "Set warmup_ratio or warmup_steps "
                    "if you are using scheduler")
        
        if self.save_model or self.save_tensors:
            assert self.save_dir is not None, (
                "Set save_dir if you are saving model and/or predictions")
        
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
        self._log(f"Starting {message}")
        yield
        time_taken = time.time() - start_time
        time_taken_str = self._convert_float_seconds_to_time_string(time_taken)
        self._log(f"{message} done, time taken = {time_taken_str}")

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
        """Save the tensors returned from inference to directory.

        Args:
            directory: Directory where the tensors will be saved.
            tensors: Dictionary of tensor name to tensor.
        """
        for name, pt in tensors.items():
            if self.save_tensors_names is None or name in self.save_tensors_names:
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
                batch_loss = self.model(**batch)
            self.accelerator.backward(batch_loss)
            if self.optimizer.gradient_state.sync_gradients and (
               self.max_grad_norm is not None):
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.use_scheduler and (
               not self.accelerator.optimizer_step_was_skipped):
                self.scheduler.step()
        return batch_loss.detach().item()

    def run(self):
        best_dev_score = None
        best_epoch = None
        epochs_left = self.patience
        save = self.save_model or self.save_tensors
        
        # Accelerate model, dataloaders, and optimizer
        (self.model, self.train_dataloader, self.dev_dataloader,
         self.optimizer) = (self.accelerator.prepare(
            self.model, self.train_dataloader, self.dev_dataloader, 
            self.optimizer))

        # Log number of training and inference batches, 
        # and number of training steps
        n_train_batches = len(self.train_dataloader)
        n_dev_batches = len(self.dev_dataloader)
        effective_train_batch_size = round(
            self.n_training_samples/n_train_batches)
        effective_dev_batch_size = round(self.n_dev_samples/n_dev_batches)
        n_training_steps = self.max_epochs * n_train_batches
        self._log("Effective train batch size = "
                        f"{effective_train_batch_size}")
        self._log("Effective dev batch size = "
                        f"{effective_dev_batch_size}")
        self._log(f"Number of training batches = {n_train_batches}")
        self._log(f"Number of inference batches = {n_dev_batches}")
        self._log(f"Number of training steps = {n_training_steps}")

        # Initialize and accelerate scheduler
        if self.use_scheduler:
            n_warmup_steps = self.warmup_steps if (
                self.warmup_steps is not None) else (
                    int(self.warmup_ratio * n_training_steps))
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=n_warmup_steps,
                num_training_steps=n_training_steps)
            self.scheduler = self.accelerator.prepare_scheduler(self.scheduler)
            self._log(f"Number of warmup steps = {n_warmup_steps}")
        
        # Training and evaluation loop
        with self._timer("training"):
            for epoch in range(self.max_epochs):
                
                if save:
                    # Create epoch directories
                    epoch_dir = os.path.join(
                        self.save_dir, f"epoch_{epoch + 1}")
                    epoch_dev_dir = os.path.join(epoch_dir, "dev")
                    os.makedirs(epoch_dir, exist_ok=True)
                    os.makedirs(epoch_dev_dir, exist_ok=True)

                # Training for one epoch
                with self._timer(f"epoch {epoch + 1} training"):
                    self.model.train()
                    running_batch_loss = []
                    running_batch_train_time = []
                    
                    # Batch training loop
                    for i, batch in enumerate(self.train_dataloader):
                        batch_start_time = time.time()
                        # One training step
                        batch_loss = self._training_step(batch)
                        batch_time_taken = time.time() - batch_start_time
                        running_batch_loss.append(batch_loss)
                        running_batch_train_time.append(batch_time_taken)

                        # Log after log_batch_frequency batches
                        if (i + 1) % self.log_batch_frequency == 0:
                            average_batch_loss = np.mean(running_batch_loss)
                            average_batch_train_time = np.mean(
                                running_batch_train_time)
                            estimated_time_remaining = (
                                self._convert_float_seconds_to_time_string(
                                average_batch_train_time * (
                                    n_train_batches-i-1)))
                            average_batch_train_time_str = (
                                self._convert_float_seconds_to_time_string(
                                average_batch_train_time))
                            self._log(f"Batch {i + 1}")
                            self._log(
                                "Average training loss @ batch = "
                                f"{average_batch_loss:.4f}")
                            self._log(
                                "Average training time taken @ batch = "
                                f"{average_batch_train_time_str}")
                            self._log(
                                "Estimated training time remaining for epoch "
                                f"{epoch + 1} = {estimated_time_remaining}")
                            running_batch_loss = []
                            running_batch_train_time = []

                # Wait for all process to complete
                self.accelerator.wait_for_everyone()

                # Save model
                if self.save_model:
                    self._log(f"Saving model after epoch {epoch + 1}")
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    self._save_model(unwrapped_model, epoch_dir)

                # Inference and evaluation on training set
                if self.evaluate_train:
                    with self._timer(
                        f"epoch {epoch + 1} training inference and evaluation"):
                        train_inference_output = self._infer(
                            self.train_dataloader, self.model)
                        train_metric = self.evaluate(**train_inference_output)
                        self._log(
                            f"Training Performance = {train_metric.score}")
                    self.accelerator.wait_for_everyone()
            
                # Inference and evaluation on dev set
                with self._timer(
                    f"epoch {epoch + 1} dev inference and evaluation"):
                    dev_inference_output = self._infer(
                        self.dev_dataloader, self.model)
                    dev_metric = self.evaluate(**dev_inference_output)
                    self._log(f"Dev Performance = {dev_metric}")
                self.accelerator.wait_for_everyone()
                if self.save_tensors:
                    self._log(
                        f"Saving dev tensors after epoch {epoch + 1}")
                    self._save_tensors(epoch_dev_dir, **dev_inference_output)

                # Early-stopping
                self._log("Checking for early-stopping")
                dev_score = dev_metric.score
                if best_dev_score is None or dev_score > best_dev_score:
                    epochs_left = self.patience
                    best_epoch = epoch + 1
                    if best_dev_score is not None:
                        delta = 100 * (dev_score - best_dev_score)
                        self._log(f"Dev score improved by {delta:.1f}")
                    best_dev_score = dev_score
                else:
                    epochs_left -= 1
                    delta = 100 * (best_dev_score - dev_score)
                    self._log(
                        f"Dev score is {delta:.1f} lower than best Dev score "
                        f"({100*best_dev_score:.1f})")
                    self._log(
                        f"{epochs_left} epochs left until Dev score to improve to"
                        " avoid early-stopping!")
                if epochs_left == 0:
                    self._log("Early stopping!")
                    break

                self._log(f"Epoch {epoch + 1} done")

        self._log(f"Best Dev score = {100*best_dev_score:.1f}")
        self._log(f"Best epoch = {best_epoch}")
    
    def _infer(self, dataloader: DataLoader, model: nn.Module) -> (
        dict[str, torch.Tensor]):
        """Run inference on the dataloader.
        Args:
            dataloader: PyTorch dataloader.
            model: PyTorch module.
        
        Returns:
            Labels and predictions tensors.
        """
        # Initialize variables
        model.eval()
        tensors: dict[str, list[torch.Tensor]] = collections.defaultdict(list)
        n_batches = len(dataloader)
        self._log(f"Number of inference batches = {n_batches}")

        # Inference Loop
        with self._timer("inference"), torch.no_grad():
            running_batch_times = []
            for i, batch in enumerate(dataloader):

                # One inference step
                start_time = time.time()
                batch_logits = model(**batch)
                batch_logits = self.accelerator.gather_for_metrics(batch_logits)
                batch = self.accelerator.gather_for_metrics(batch)
                batch["logits"] = batch_logits
                for name, tensor in batch.items():
                    tensors[name].append(tensor)
                time_taken = time.time() - start_time
                running_batch_times.append(time_taken)

                # Log after log_batch_frequency batches
                if (i + 1) % self.log_batch_frequency == 0:
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

                    self._log(f"Batch {i + 1}")
                    self._log("Average inference time @ batch = "
                                f"{average_time_per_batch_str}")
                    self._log("Estimated inference time remaining = "
                                f"{estimated_time_remaining_str}")

        # Concat tensors
        output: dict[str, torch.Tensor] = {}
        for name, tensor_list in tensors.items():
            output[name] = torch.cat(tensor_list, dim=0)
        return output
    
    def evaluate(self, **tensors) -> Metric:
        """Evaluate the output of inference"""
        raise NotImplementedError