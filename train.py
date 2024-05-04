import os
import torch
from tqdm.notebook import tqdm
from typing import List, Tuple
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np


from model import Transformer
from util import Translator, load_checkpoint, plot_losses, save_checkpoint


def train_epoch(
    train_loader: DataLoader,
    epoch: int,
    num_epochs: int,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    clip_grad: float = 1.0,
) -> Tuple[float, float]:
    model.train()

    current_loss = 0

    for batch in tqdm(train_loader, desc=f"Training, epoch {epoch}/{num_epochs}"):
        optimizer.zero_grad()

        src = batch["src"].to(device)
        trg = batch["trg"].to(device)
        src_mask = batch["src_mask"].to(device)
        trg_mask = batch["trg_mask"].to(device)

        # (B, T, vocab_size), predictions for every token
        pred = model(src, trg, src_mask, trg_mask)

        # (B * (T - 1), vocab_size), predictions for every token except for last
        pred = pred[:, :-1, :].reshape(-1, pred.shape[2])

        # (B * (T - 1)), starting with second token
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(pred, trg)

        loss.backward()

        clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        current_loss += loss.item()

        loss = None

    total_loss = current_loss / len(train_loader)

    return total_loss, np.exp(total_loss)


def validate_epoch(
    val_loader: DataLoader,
    epoch: int,
    num_epochs: int,
    model: Transformer,
    criterion: torch.nn.Module,
    device,
) -> Tuple[float, float]:
    current_loss = 0
    model.eval()
    for batch in tqdm(val_loader, desc=f"Validating, epoch {epoch}/{num_epochs}"):
        src = batch["src"].to(device)
        trg = batch["trg"].to(device)
        src_mask = batch["src_mask"].to(device)
        trg_mask = batch["trg_mask"].to(device)

        # (B, T, vocab_size), predictions for every token
        pred = model(src, trg, src_mask, trg_mask)

        # (B * (T - 1), vocab_size), predictions for every token except for last
        pred = pred[:, :-1, :].reshape(-1, pred.shape[2])

        # (B * (T - 1)), starting with second token
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(pred, trg)

        current_loss += loss.item()

        loss = None

    total_loss = current_loss / len(val_loader)

    return total_loss, np.exp(total_loss)


def train(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    num_epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    clip_grad: float = 1.0,
    path_to_save: str = "model/training.pt",
    path_to_save_best: str = "model/best.pt",
    loss_label: str = "loss",
    metric_label: str = "metric",
    translator: Translator | None = None,
    examples_to_translate: List[str] = [],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    train_loss, train_metric = [], []
    val_loss, val_metric = [], []
    last_epoch = 0

    if os.path.exists(path_to_save):
        checkpoint = load_checkpoint(path_to_save)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        train_loss, train_metric, val_loss, val_metric = (
            checkpoint["train_loss"],
            checkpoint["train_metric"],
            checkpoint["valid_loss"],
            checkpoint["valid_metric"],
        )
        last_epoch = checkpoint["epoch"]
        plot_losses(
            train_loss,
            train_metric,
            val_loss,
            val_metric,
            loss_label=loss_label,
            metric_label=metric_label,
        )
        if scheduler is not None:
            print(f"LR: {scheduler.get_last_lr()}")
        if translator is not None:
            print("Translating examples:")
            for i, s in enumerate(examples_to_translate):
                translated = translator.translate(s)
                print(f"{i + 1}. {s}")
                print(translated)
                print("****************")

    device = model.device
    for epoch in range(last_epoch + 1, num_epochs + 1):
        loss_value, metric_value = train_epoch(
            train_loader,
            epoch,
            num_epochs,
            model,
            optimizer,
            criterion,
            device,
            clip_grad=clip_grad,
        )
        train_loss.append(loss_value)
        train_metric.append(metric_value)

        with torch.no_grad():
            loss_value, metric_value = validate_epoch(
                val_loader, epoch, num_epochs, model, criterion, device
            )

        val_loss.append(loss_value)
        val_metric.append(metric_value)

        plot_losses(
            train_loss,
            train_metric,
            val_loss,
            val_metric,
            loss_label=loss_label,
            metric_label=metric_label,
        )

        if scheduler is not None:
            scheduler.step()
            print(f"LR: {scheduler.get_last_lr()}")

        print("Saving checkpoint...")
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            train_loss,
            train_metric,
            val_loss,
            val_metric,
            path=path_to_save,
        )

        if val_metric[-1] == min(val_metric):
            print("Saving checkpoint as best...")
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                train_loss,
                train_metric,
                val_loss,
                val_metric,
                path=path_to_save_best,
            )

        if translator is not None:
            print("Translating examples:")
            for i, s in enumerate(examples_to_translate):
                translated = translator.translate(s)
                print(f"{i + 1}. {s}")
                print(translated)
                print("****************")

    return train_loss, train_metric, val_loss, val_metric
