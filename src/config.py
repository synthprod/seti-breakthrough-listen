from datetime import datetime
from pytz import timezone
import torch

class config:
    n_fold = 5
    epochs = 20
    image_size = (1024, 1024) # default: (256, 1638) 273 * 6 = 1638
    clip_rate = 3.5
    do_pseudo_labeling = False
    is_metric_learning = False
    use_amp = True
    freeze = False
    early_stop = False
    model_name = "tf_efficientnetv2_b0"
    training_date = str(datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d'))
    weight_path = "weights/" + training_date
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    gradient_accumulation_steps = 1
    num_classes = 1
    lr = 5e-4
    num_workers = 4
    do_mixup = True
    alpha = 0.4 # mixup
    log_file_name = "logs/" + str(datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d')) + ".log"
