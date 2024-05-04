from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import clear_output
from transformers import BertTokenizer, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple
import random
from tokenizers.processors import TemplateProcessing
from datasets import DatasetDict

from model import Transformer


class Translator:
    def __init__(
        self,
        model: Transformer,
        tokenizer_src: BertTokenizer,
        tokenizer_trg: BertTokenizer,
    ) -> None:
        self.model = model
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg

    def translate(
        self,
        s: str,
        skip_special_tokens: bool = True,
    ) -> str:
        self.model.eval()

        input = self.tokenizer_src.encode(s, return_tensors="pt").to(self.model.device)

        output = self.model.inference(input)

        return self.tokenizer_trg.decode(
            output, skip_special_tokens=skip_special_tokens
        )


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    train_loss: List[float],
    train_metric: List[float],
    valid_loss: List[float],
    valid_metric: List[float],
    path: str = "model-data/translator.pth",
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "train_metric": train_metric,
        "valid_loss": valid_loss,
        "valid_metric": valid_metric,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
) -> dict:
    return torch.load(path)


def plot_losses(
    train_loss: List[float],
    train_metric: List[float],
    val_loss: List[float],
    val_metric: List[float],
    loss_label: str = "loss",
    metric_label: str = "metric",
):
    clear_output()
    n_epochs = len(train_loss)
    epochs = np.arange(1, n_epochs + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.lineplot(
        x=epochs, y=train_loss, marker="o", label=f"train {loss_label}", ax=ax[0]
    )
    sns.lineplot(x=epochs, y=val_loss, marker="o", label=f"val {loss_label}", ax=ax[0])
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel(loss_label)
    ax[0].set_title(f"epoch -- {loss_label}")

    sns.lineplot(
        x=epochs, y=train_metric, marker="o", label=f"train {metric_label}", ax=ax[1]
    )
    sns.lineplot(
        x=epochs, y=val_metric, marker="o", label=f"val {metric_label}", ax=ax[1]
    )
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel(metric_label)
    ax[1].set_title(f"epoch -- {metric_label}")

    ax[0].legend()
    ax[1].legend()
    plt.show()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dataloaders_lazy(
    dataset_dict: DatasetDict,
    batch_size: int,
    tokenizer_src: BertTokenizer,
    tokenizer_trg: BertTokenizer,
):
    def collate_fn(batch: dict) -> dict:
        src_encoded = tokenizer_src.batch_encode_plus(
            [x["src"] for x in batch], return_tensors="pt", padding=True
        )
        trg_encoded = tokenizer_trg.batch_encode_plus(
            [x["trg"] for x in batch], return_tensors="pt", padding=True
        )

        trg_encoded["attention_mask"] = torch.tril(trg_encoded["attention_mask"])

        return {
            "src": src_encoded["input_ids"],
            "trg": trg_encoded["input_ids"],
            "src_mask": src_encoded["attention_mask"],
            "trg_mask": trg_encoded["attention_mask"],
            "trg_text": [x["trg"] for x in batch],  # for BLEU
        }

    return (
        DataLoader(
            dataset_dict["train"],
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
        ),
        DataLoader(
            dataset_dict["valid"],
            batch_size=batch_size,
            collate_fn=collate_fn,
        ),
        DataLoader(
            dataset_dict["test"],
            batch_size=batch_size,
            collate_fn=collate_fn,
        ),
    )


def xavier_init(module: torch.nn.Module):
    for p in module.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)


def add_preprocessing(tokenizer: AutoTokenizer) -> None:
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=tokenizer.bos_token + " $A " + tokenizer.eos_token,
        special_tokens=[
            (tokenizer.eos_token, tokenizer.eos_token_id),
            (tokenizer.bos_token, tokenizer.bos_token_id),
        ],
    )


def get_tokenizers() -> Tuple[BertTokenizer, BertTokenizer]:
    tokenizer_src = AutoTokenizer.from_pretrained(
        "deepvk/bert-base-uncased",
        bos_token="<s>",
        add_bos_token=True,
        add_eos_token=True,
        eos_token="</s>",
    )

    tokenizer_trg = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-uncased",
        bos_token="<s>",
        add_bos_token=True,
        add_eos_token=True,
        eos_token="</s>",
    )

    add_preprocessing(tokenizer_src)
    add_preprocessing(tokenizer_trg)

    return tokenizer_src, tokenizer_trg


def count_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
