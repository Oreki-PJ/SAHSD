import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn.metrics import precision_score
from argparse import ArgumentParser
from transformers import AutoModelForMaskedLM, AutoTokenizer, set_seed

from src.utils import load_config, get_logger, get_optimizer_scheduler, compute_metrics, compute_metrics_multi
from src.data import get_data_reader, get_data_loader
from src.model import get_pet_mappers

import pdb
import pandas as pd


def generate_label(model, pet, config, dataloader):
    all_labels, all_preds = [], []

    model.eval()
    test_loss = 0.
    for batch in tqdm(dataloader, desc=f'[test]'):
        with torch.no_grad():
            pet.forward_step(batch)
            loss = pet.get_loss(batch, config.full_vocab_loss)
            test_loss += loss.item()
        all_preds.append(pet.get_predictions(batch))
        # all_labels.append(batch["label_ids"])
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()

    return all_preds



def test(config, **kwargs):
    config.update(kwargs)
    logger = get_logger('test')
    logger.info(config)
    device = torch.device('cuda:0' if config.use_gpu else 'cpu')
    logger.info(f' * * * * * Testing * * * * *')

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(config.trained_model)
    model = AutoModelForMaskedLM.from_pretrained(config.trained_model)
    model.to(device)

    # Load data
    reader = get_data_reader(config.task_name)
    train_loader = get_data_loader(reader, config.train_path + f'{config.train_sample_nums}/' + f'{config.seed}.csv', 'train',
                                   tokenizer, config.max_seq_len, config.train_batch_size, device, config.shuffle)
    dev_loader = get_data_loader(reader, config.dev_path, 'dev',
                                 tokenizer, config.max_seq_len, config.test_batch_size, device)
    test_loader = get_data_loader(reader, config.test_path, 'test',
                                  tokenizer, config.max_seq_len, config.test_batch_size, device)
    pet = get_pet_mappers(tokenizer, reader, model, device,
                             config.pet_method)

    train_preds = generate_label(model, pet, config, train_loader)
    if not (os.path.exists('./data/SBIC/dev_data_pp50_with_senti_pre.csv')) or \
          not (os.path.exists('./data/SBIC/test_data_pp50_with_senti_pre.csv')):
        dev_preds = generate_label(model, pet, config, dev_loader)
        test_preds = generate_label(model, pet, config, test_loader)

    train_data = pd.read_csv(config.train_path + f'{config.train_sample_nums}/' + f'{config.seed}.csv', index_col=0)
    dev_data = pd.read_csv("./data/SBIC/dev_data.csv", index_col=0)
    test_data = pd.read_csv("./data/SBIC/test_data.csv", index_col=0)
    train_data["senti_pre"] = train_preds
    if not (os.path.exists('./data/SBIC/dev_data_pp50_with_senti_pre.csv')) or \
          not (os.path.exists('./data/SBIC/test_data_pp50_with_senti_pre.csv')):
        dev_data["senti_pre"] = dev_preds
        test_data["senti_pre"] = test_preds

    train_data.to_csv(config.train_path + f'{config.train_sample_nums}/' + f'{config.seed}_pp50_with_senti_pre.csv')
    if not (os.path.exists('./data/SBIC/dev_data_pp50_with_senti_pre.csv')) or \
          not (os.path.exists('./data/SBIC/test_data_pp50_with_senti_pre.csv')):
        dev_data.to_csv('./data/SBIC/dev_data_pp50_with_senti_pre.csv')
        test_data.to_csv('./data/SBIC/test_data_pp50_with_senti_pre.csv')

    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config/origin.yml',
                        help='Configuration file storing all parameters')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--seed', type=int, required=True, help='Random seed.')
    parser.add_argument('--train_sample_nums', type=int, required=True, help='train sample nums')
    args = parser.parse_args()

    assert args.do_train or args.do_test, f'At least one of do_train or do_test should be set.'
    cfg = load_config(args.config)
    # os.makedirs(cfg.output_dir, exist_ok=True)

    # fix seed and train sample nums
    cfg.seed = args.seed
    cfg.train_sample_nums = args.train_sample_nums

    # if args.do_train:
    #     train(cfg)
    if args.do_test:
        test(cfg)
