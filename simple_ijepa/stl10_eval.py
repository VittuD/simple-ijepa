# simple_ijepa/stl10_eval.py

import logging
from typing import Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from simple_ijepa.utils import inference_transforms

import warnings

warnings.filterwarnings("ignore")


def logistic_regression(
    embeddings,
    labels,
    embeddings_val,
    labels_val,
    logger: Optional[logging.Logger] = None,
    wandb_run: Optional[Any] = None,
    step: Optional[int] = None,
    prefix: str = "eval",
):
    X_train, X_test = embeddings, embeddings_val
    y_train, y_test = labels, labels_val

    clf = LogisticRegression(max_iter=100)
    clf = CalibratedClassifierCV(clf)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    if logger is None:
        logger = logging.getLogger("simple_ijepa.stl10_eval")

    acc = accuracy_score(y_test, y_pred)
    logger.info("Accuracy STL10 (%s): %.4f", prefix, acc)

    # Optional W&B logging
    if wandb_run is not None:
        try:
            wandb_run.log({f"{prefix}/accuracy": acc}, step=step)
        except Exception:
            if logger is not None:
                logger.warning("Failed to log STL10 accuracy to wandb.")


class STL10Eval:
    def __init__(
        self,
        image_size: int = 96,
        dataset_path: str = "data/",
        logger: Optional[logging.Logger] = None,
        wandb_run: Optional[Any] = None,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.logger = logger or logging.getLogger("simple_ijepa.stl10_eval")
        self.wandb_run = wandb_run

        transform = inference_transforms(img_size=(image_size, image_size))
        train_ds = torchvision.datasets.STL10(
            root=dataset_path,
            split="train",
            transform=transform,
            download=True,
        )
        val_ds = torchvision.datasets.STL10(
            root=dataset_path,
            split="test",
            transform=transform,
            download=True,
        )

        self.train_loader = DataLoader(train_ds, batch_size=64, num_workers=2)
        self.val_loader = DataLoader(val_ds, batch_size=64, num_workers=2)

    @torch.inference_mode
    def evaluate(self, ijepa_model, global_step: Optional[int] = None, prefix: str = "eval"):
        model = ijepa_model.target_encoder
        # model = ijepa_model.context_encoder
        embeddings, labels = self._get_image_embs_labels(
            model, self.train_loader
        )
        embeddings_val, labels_val = self._get_image_embs_labels(
            model, self.val_loader
        )
        logistic_regression(
            embeddings,
            labels,
            embeddings_val,
            labels_val,
            logger=self.logger,
            wandb_run=self.wandb_run,
            step=global_step,
            prefix=prefix,
        )

    @torch.inference_mode
    def _get_image_embs_labels(self, model, dataloader):
        embs, labels = [], []
        for _, (images, targets) in enumerate(dataloader):
            with torch.no_grad():
                images = images.to(self.device)
                out = model(images)
                features = out.cpu().detach()
                features = features.mean(dim=1)
                embs.extend(features.tolist())
                labels.extend(targets.cpu().detach().tolist())
        return np.array(embs), np.array(labels)
