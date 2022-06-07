import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch_optimizer
import mlflow
from pywick.losses import BinaryFocalLoss

# local files
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.utils import *
from src.models import get_model
from src.config import config
from src.trainer import Trainer

# setting
score = []
train_df = pd.read_csv("train_labels.csv")
print(train_df.shape)
logger = setup_logger(config.log_file_name)
seed_torch(cudnn_benchmark=False)


rates = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

for clip_rate in rates:
    logger.info(f"clip rate: {clip_rate}")

    _train_df = train_df.query(f"fold != 0").reset_index(drop=True)
    _val_df = train_df.query(f"fold == 0").reset_index(drop=True)
    
    model = get_model(config.model_name, config.device, pretrained=True, num_classes=config.num_classes)
    dataloaders_dict = get_dataloaders_dict(_train_df, _val_df, clip_rate)
    criterion = BinaryFocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = None
    
    trainer = Trainer(model, dataloaders_dict, optimizer, criterion, scheduler=scheduler, logger=logger, do_mixup=config.do_mixup)
    trainer.not_save_model() # not save model
    trainer.not_save_result_image() # not save result image
    trainer.set_n_fold(0)
    best_score = trainer.fit()
    logger.info(f"clip rate {clip_rate} best score: {best_score:.4}")
    score.append(best_score)

logger.debug(score)

# mlflow
with mlflow.start_run():
    params = {
        "input size": config.image_size,
        "mixup": config.do_mixup,
        "model_name": config.model_name,
        "batch_size": config.batch_size,
        "loss_function": criterion.__class__.__name__,
        "focal_loss_gamma": 1.333,
        "focal_loss_alpha": 1.0,
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": config.lr,
        "scheduler": scheduler,
        "epochs": config.epochs,
        "early_stop": config.early_stop,
        "remarks": "clip rate test",
    }
    mlflow.log_params(params)

    for clip_rate, score in zip(rates, score):
        mlflow.log_metric(f"clip rate {clip_rate}", score)

mlflow.end_run()