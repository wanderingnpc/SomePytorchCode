import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import time
import logging
from config import Config
from dataset import load_dataset_for_mlm, load_dataset_for_mlm_cl
from train import training_mlm, training_mlm_cl
from utils import set_logger

config = Config()

os.makedirs('./log', exist_ok=True)
set_logger(config.log_save_path + '_' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) + '.txt')
for name, value in vars(config).items():
    logging.info('$$$$$ custom para {}: {}'.format(name, value))

if config.contrastive:
    dataset = load_dataset_for_mlm_cl(config)
    config.train_dataloader = dataset.train_dataloader
    config.eval_dataloader = dataset.eval_dataloader
    training_mlm_cl(config)
else:
    dataset = load_dataset_for_mlm(config)
    config.train_dataloader = dataset.train_dataloader
    config.eval_dataloader = dataset.eval_dataloader
    training_mlm(config)
