"""
Training callbacks for MicroFeatEX.

Provides an extensible callback system to hook into the training loop
without modifying the core Trainer class.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from microfeatex.training.trainer import Trainer


class TrainerCallback(ABC):
    """Base class for training callbacks.

    Override any of the hook methods to add custom behavior.
    All hooks receive the trainer instance for access to state.
    """

    def on_train_begin(self, trainer: Trainer) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        """Called at the end of training."""
        pass

    def on_step_begin(self, trainer: Trainer, step: int) -> None:
        """Called before each training step."""
        pass

    def on_step_end(
        self,
        trainer: Trainer,
        step: int,
        loss: float,
        metrics: dict[str, Any],
    ) -> None:
        """Called after each training step.

        Args:
            trainer: The Trainer instance.
            step: Current training step.
            loss: Loss value for this step.
            metrics: Dictionary of metrics for this step.
        """
        pass

    def on_checkpoint(self, trainer: Trainer, step: int) -> None:
        """Called when a checkpoint is saved."""
        pass


class LoggingCallback(TrainerCallback):
    """Handles TensorBoard logging.

    Logs metrics to TensorBoard at specified intervals.
    """

    def __init__(self, log_interval: int = 10, param_log_interval: int = 100):
        """Initialize logging callback.

        Args:
            log_interval: Steps between metric logging.
            param_log_interval: Steps between hyperparameter logging.
        """
        self.log_interval = log_interval
        self.param_log_interval = param_log_interval

    def on_step_end(
        self,
        trainer: Trainer,
        step: int,
        loss: float,
        metrics: dict[str, Any],
    ) -> None:
        """Log metrics to TensorBoard."""
        if step % self.log_interval == 0:
            for key, value in metrics.items():
                trainer.writer.add_scalar(key, value, step)

        if step % self.param_log_interval == 0:
            trainer.writer.add_scalar(
                "params/conf_thresh", trainer.hp["conf_thresh"], step
            )
            trainer.writer.add_scalar("params/w_fine", trainer.hp["w_fine"], step)


class VisualizationCallback(TrainerCallback):
    """Handles training visualizations.

    Logs visual outputs (heatmaps, matches, etc.) to TensorBoard.
    """

    def __init__(self, vis_interval: int = 500):
        """Initialize visualization callback.

        Args:
            vis_interval: Steps between visualizations.
        """
        self.vis_interval = vis_interval
        self._last_vis_data: tuple | None = None

    def store_vis_data(self, vis_data: tuple) -> None:
        """Store visualization data from training step.

        Call this from the training loop to provide data for visualization.
        """
        self._last_vis_data = vis_data

    def on_step_end(
        self,
        trainer: Trainer,
        step: int,
        loss: float,
        metrics: dict[str, Any],
    ) -> None:
        """Generate and log visualizations."""
        if step % self.vis_interval != 0 or self._last_vis_data is None:
            return

        p1, p2, out1, t_out1 = self._last_vis_data
        t_heat = t_out1.get("heatmap", None)

        trainer.vis.log_advanced_visuals(
            step=step,
            img1=p1,
            img2=p2,
            s_heat=out1["heatmap"],
            t_heat=t_heat,
            desc1=out1["descriptors"],
            desc2=out1["descriptors"],
            s_rel=out1["reliability"],
        )


class GradientMonitorCallback(TrainerCallback):
    """Monitors gradient statistics.

    Logs gradient norms and detects anomalies.
    """

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.nan_count = 0

    def on_step_end(
        self,
        trainer: Trainer,
        step: int,
        loss: float,
        metrics: dict[str, Any],
    ) -> None:
        """Log gradient statistics."""
        if step % self.log_interval == 0:
            # Compute gradient norm manually for monitoring
            total_norm = 0.0
            for p in trainer.student.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5

            trainer.writer.add_scalar("debug/computed_grad_norm", total_norm, step)

            if not torch.isfinite(torch.tensor(total_norm)):
                self.nan_count += 1
                trainer.writer.add_scalar(
                    "debug/nan_gradient_count", self.nan_count, step
                )


class CallbackHandler:
    """Manages multiple callbacks.

    Provides a single interface to invoke all registered callbacks.
    """

    def __init__(self, callbacks: list[TrainerCallback] | None = None):
        self.callbacks = callbacks or []

    def add_callback(self, callback: TrainerCallback) -> None:
        """Add a callback to the handler."""
        self.callbacks.append(callback)

    def on_train_begin(self, trainer: Trainer) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer: Trainer) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_step_begin(self, trainer: Trainer, step: int) -> None:
        for cb in self.callbacks:
            cb.on_step_begin(trainer, step)

    def on_step_end(
        self,
        trainer: Trainer,
        step: int,
        loss: float,
        metrics: dict[str, Any],
    ) -> None:
        for cb in self.callbacks:
            cb.on_step_end(trainer, step, loss, metrics)

    def on_checkpoint(self, trainer: Trainer, step: int) -> None:
        for cb in self.callbacks:
            cb.on_checkpoint(trainer, step)
