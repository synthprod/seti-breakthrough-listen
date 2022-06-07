import pandas as pd
import torch
from tqdm.auto import tqdm as tqdm

from sklearn.metrics import roc_auc_score

# local files
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import config as config
from src.utils import mixup_data, get_dataloader, get_dataloader_for_no_da
from src.metric_recorder import MetricRecorder


class Trainer:
    def __init__(self, model, dataloaders_dict, optimizer, criterion, scheduler=None, logger=None, do_mixup=False, do_pseudo_labeling=False):
        # 初期値設定
        self.best_loss = 10**5
        self.best_score = 0.0
        self.counter = 0 # early_stopまでのカウンター
        self.early_stop_limit = 3 # スコア更新の失敗の回数でearly stopの実施を決める
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        self.n_fold = int()

        # setter
        self.model = model
        self.train_loader = dataloaders_dict["train"]
        self.valid_loader = dataloaders_dict["val"]
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        if logger is None:
            raise ValueError("logger is NoneType.")
        else:
            self.logger = logger

        self.freezed = True if config.freeze else False
        if self.freezed:
            self.logger.info("Currently the model is frozen.")
        self.do_mixup = do_mixup
        self.do_pseudo_labeling = do_pseudo_labeling
        self.is_save = True
        self.save_result_image = True

        self.metric_recorder = MetricRecorder(config.epochs)

    def set_n_fold(self, n_fold):
        self.n_fold = n_fold

    # train
    def fit(self):
        for epoch in range(config.epochs):
            self.logger.info(f"Epoch {epoch+1} / {config.epochs}")

            # 4epochから解除 config.freeze = Trueでのみ適用
            # if self.freezed is True and epoch + 1 == 4:
            #     self.logger.info(f"unfreeze model..")
            #     self._unfreeze_model()
            #     self.freezed = False

            # ラスト3epochでDAなしにする
            # if epoch + 1 == config.epochs - 3:
            #     self.logger.info(f"start learning without data augmentation..")
            #     self._prepare_for_last_3_epochs()

            for phase in ["train", "val"]:
                if phase == "train":
                    if self.do_mixup:
                        epoch_loss = self._train_with_mixup()
                    else:
                        epoch_loss = self._train()
                    self.metric_recorder.record_score_when_training(epoch_loss)
                    self.logger.info(f'fold - {self.n_fold}:: phase: {phase}, loss: {epoch_loss:.4f}')
                else:
                    epoch_loss, epoch_score = self._valid()
                    self.metric_recorder.record_score_when_validating(epoch_loss, epoch_score)
                    self.logger.info(f'fold - {self.n_fold}:: phase: {phase}, loss: {epoch_loss:.4f}, roc_auc: {epoch_score:.4f}, -- learning rate: {self.optimizer.param_groups[0]["lr"]}')
                    self._update_score(epoch_loss, epoch_score)

                    if self.do_pseudo_labeling and not self.freezed:
                        self.logger.info(f'create pseudo labeling..')
                        self._pseudo_labeling()

            if config.early_stop and self.counter > self.early_stop_limit:
                self.logger.debug("early stopping..")
                break

        if self.save_result_image:
            self.metric_recorder.save_as_image(file_name=f"fold_{self.n_fold}.png")

        return self.best_score


    def _train(self):
        self.model.train()
        epoch_loss = 0.0

        for idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            batch_size = inputs.shape[0]

            if config.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.view(-1), labels)
                    if config.gradient_accumulation_steps > 1:
                        loss = loss / config.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()

                    if (idx + 1) % config.gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps
                loss.backward()

                if (idx + 1) % config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if torch.isnan(loss):
                raise ValueError("contains the loss value of nan")

            epoch_loss += loss.item() * batch_size
            
        epoch_loss = epoch_loss / len(self.train_loader.dataset)

        return epoch_loss


    def _train_with_mixup(self):
        self.model.train()
        epoch_loss = 0.0

        for idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            batch_size = inputs.shape[0]

            if config.use_amp:
                with torch.cuda.amp.autocast():
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, config.alpha)
                    inputs, targets_a, targets_b = inputs.to(config.device), targets_a.to(config.device), targets_b.to(config.device)
                    outputs = self.model(inputs)
                    outputs = outputs.view(-1)
                    loss = self.criterion(outputs, targets_a) * lam + self.criterion(outputs, targets_b) * (1. - lam)

                    if config.gradient_accumulation_steps > 1:
                        loss = loss / config.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()

                    if (idx + 1) % config.gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
            else:
                raise NotImplementedError("try use amp = True")

            if torch.isnan(loss):
                raise ValueError("contains the loss value of nan")

            epoch_loss += loss.item() * batch_size
            
        epoch_loss = epoch_loss / len(self.train_loader.dataset)

        return epoch_loss



    def _valid(self):
        self.model.eval()
        epoch_loss, epoch_score = float(), float()
        y_pred, y_true = [], []

        with torch.no_grad():
            for _, (inputs, labels) in enumerate(self.valid_loader):
                inputs = inputs.to(config.device)
                labels = labels.to(config.device)
                batch_size = inputs.shape[0]

                if config.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                loss = self.criterion(outputs.view(-1), labels)
                if torch.isnan(loss):
                    raise ValueError("contains the loss value of nan")

                epoch_loss += loss.item() * batch_size

                y_pred.extend(outputs.sigmoid().to("cpu").numpy())
                y_true.extend(labels.to("cpu").numpy())

        epoch_loss = epoch_loss / len(self.valid_loader.dataset)
        epoch_score = roc_auc_score(y_true, y_pred)

        if self.scheduler is not None:
            self.scheduler.step()

        return epoch_loss, epoch_score


    def _update_score(self, epoch_loss, epoch_score):
        if self.best_score <= epoch_score:
            self.best_score = epoch_score
            self.best_loss = epoch_loss
            self.logger.debug(f"update best score: {self.best_score:.4f}")

            if self.is_save:
                torch.save(self.model.state_dict(), f"{config.weight_path}/{config.model_name}_fold{self.n_fold}.pth")
            self.counter = 0
        
        elif self.best_loss >= epoch_loss:
            self.best_loss = epoch_loss
            self.counter = 0

        else:
            self.logger.debug("There is no update of the best score")
            self.counter += 1


    def _unfreeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = True


    def not_save_model(self):
        self.is_save = False


    def not_save_result_image(self):
        self.save_result_image = False


    def _pseudo_labeling(self):
        test_df = pd.read_csv("train_for_pseudo_labeling.csv")
        test_df = test_df.query(f"fold != {self.n_fold-1}").reset_index(drop=True)
        _test_df = test_df[test_df["fold"].isnull()].reset_index(drop=True)
        _test_loader = get_dataloader(_test_df, mode="test", batch_size=config.batch_size, clip_rate=config.clip_rate)
        
        y_pred = []
        with torch.no_grad():
            for inputs in _test_loader:
                inputs = inputs.to(config.device)

                if config.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                y_pred.extend(outputs.sigmoid().to("cpu").numpy())

        # 0, 1の値に変換する
        for idx, val in enumerate(y_pred):
            re_val = 0 if val < 0.5 else 1
            y_pred[idx] = re_val
        
        test_df.loc[48000:, "target"] = y_pred
        self.train_loader = get_dataloader(test_df, mode="train", batch_size=config.batch_size, clip_rate=config.clip_rate)


    # 残り3epochのときにDAなしで学習する用
    def _prepare_for_last_3_epochs(self):
        train_df = pd.read_csv("train_labels.csv")
        train_df = train_df.query(f"fold != {self.n_fold-1}").reset_index(drop=True)
        self.train_loader = get_dataloader_for_no_da(train_df, batch_size=config.batch_size, clip_rate=config.clip_rate)

        # pseudo labelingをOFFにする
        self.do_pseudo_labeling = False
        # mixupをOFFにする
        self.do_mixup = False