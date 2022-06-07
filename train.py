import pandas as pd
import numpy as np
import torch
import mlflow
from pywick.losses import BinaryFocalLoss
import warnings
warnings.filterwarnings("ignore")

# local files
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.utils import *
from src.models import get_model
from src.config import config
from src.trainer import Trainer


if not os.path.exists(config.weight_path):
    os.makedirs(config.weight_path, exist_ok=True)

cv_score = []
train_df = pd.read_csv("train_labels.csv")
print(train_df.shape)
logger = setup_logger(config.log_file_name)
seed_torch(cudnn_benchmark=True)


for fold in range(config.n_fold):
    logger.debug(f"fold #{fold+1}")
    _train_df = train_df.query(f"fold != {fold}").reset_index(drop=True)
    _val_df = train_df.query(f"fold == {fold}").reset_index(drop=True)
    
    logger.info(f"getting {config.model_name}.. device: {config.device}")
    model = get_model(config.model_name, config.device, pretrained=True, num_classes=config.num_classes, freeze=config.freeze)
    dataloaders_dict = get_dataloaders_dict(_train_df, _val_df, clip_rate=config.clip_rate)
    focal_loss_gamma = 1.333
    focal_loss_alpla = 1.0
    criterion = BinaryFocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    T_mult = 1
    eta_min = 1e-7
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.epochs, T_mult=T_mult, eta_min=eta_min)
    
    trainer = Trainer(model, dataloaders_dict, optimizer, criterion, scheduler=scheduler, logger=logger, do_mixup=config.do_mixup, do_pseudo_labeling=config.do_pseudo_labeling)
    trainer.set_n_fold(fold+1)
    best_score = trainer.fit()
    logger.debug(f"fold #{fold+1} best score: {best_score:.4}")
    cv_score.append(best_score)

logger.debug(cv_score)

# mlflow
with mlflow.start_run():
    params = {
        "input size": config.image_size,
        "freeze model": config.freeze,
        "pseudo labeling": config.do_pseudo_labeling,
        "clip_rate": config.clip_rate,
        "mixup": config.do_mixup,
        "model_name": config.model_name,
        "batch_size": config.batch_size,
        "loss_function": criterion.__class__.__name__,
        "focal_loss_gamma": focal_loss_gamma,
        "focal_loss_alpha": focal_loss_alpla,
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": config.lr,
        "scheduler": scheduler.__class__.__name__,
        "scheduler T_0": config.epochs,
        "scheduler T_mult": T_mult,
        "scheduler eta_min": eta_min,
        "n_fold": config.n_fold,
        "epochs": config.epochs,
        "early_stop": config.early_stop,
        "remarks": "new dataset, update image size",
    }
    mlflow.log_params(params)

    for idx, score in enumerate(cv_score):
        mlflow.log_metric(f"fold {idx+1}", score)
        mlflow.log_artifact(f"result/fold_{idx+1}.png")

    cv_score = np.mean(cv_score, axis=0)
    logger.debug(f"cv score: {cv_score:.4}")
    mlflow.log_metric("cv score", cv_score)

mlflow.end_run()
