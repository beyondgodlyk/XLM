# Taken from https://pytorch.org/text/main/tutorials/sst2_classification_non_distributed.html

from collections import OrderedDict
import time
import numpy as np
import os
import argparse
import torch
import random
import torch.nn as nn

from xlm.utils import to_cuda

from xlm.utils import bool_flag, initialize_exp

from torch.utils.data import Dataset
from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torchtext.functional as F
from torch.optim import AdamW

class SentenceDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Train XLM Roberta classifier used for evaluating TST model")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    # batch parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences per batch")
    parser.add_argument("--max_len", type=int, default=30,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--tokens_per_batch", type=int, default=-1,
                        help="Number of tokens per batch")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
    
    # model parameters
    parser.add_argument("--gelu_activation", type=bool_flag, default=True,
                        help="Use a GELU activation instead of ReLU")
    
    # output paths
    parser.add_argument("--output_path", type=str, default="", help="Output path")

    # reload checkpoint
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")

    parser.add_argument("--max_vocab", type=int, default=-1, help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--min_count", type=int, default=0, help="Minimum vocabulary count")

    # data
    parser.add_argument("--data_path", type=str, default="",
                        help="Data path")
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")

    # input sentence noise
    parser.add_argument("--word_shuffle", type=float, default=0,
                        help="Randomly shuffle input words (0 to disable)")
    parser.add_argument("--word_dropout", type=float, default=0,
                        help="Randomly dropout input words (0 to disable)")
    parser.add_argument("--word_blank", type=float, default=0,
                        help="Randomly blank input words (0 to disable)")

    # training parameters
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=100000,
                        help="Epoch size / evaluation frequency (-1 for parallel data size)")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")
        
    # float16 / AMP API
    parser.add_argument("--amp", type=int, default=-1,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")
    
    # memory parameters
    parser.add_argument("--use_memory", type=bool_flag, default=False,
                        help="Use an external memory")
    
    # debug (required for using load binarized)
    parser.add_argument("--debug_train", type=bool_flag, default=False,
                        help="Use valid sets for train sets (faster loading)")
    
    # Datasets for EN
    parser.add_argument("--use_yelp_EN_lowercase", type=bool_flag, default=True,
                        help="True if Yelp EN dataset with lowercased text is used for domain adaptive training")
    
    return parser

def check_params(params):
    """
    Check parameters and add necessary ones.
    """
    # Disabled Multi GPU for the training to finish
    params.multi_gpu = False

    # data path
    assert os.path.isdir(params.data_path), params.data_path

    # check languages
    params.langs = ['en']
    # assert sorted(params.langs) == params.langs
    params.id2lang = {k: v for k, v in enumerate(sorted(params.langs))}
    params.lang2id = {k: v for v, k in params.id2lang.items()}
    params.n_langs = len(params.langs)

    params.labels = [0, 1]

    params.xlm_classifier_train_dataset = {
        lang: {
            splt: (os.path.join(params.data_path, 'tst.%s.0.%s.tok' % (splt, lang)),
                   os.path.join(params.data_path, 'tst.%s.1.%s.tok' % (splt, lang)))
            for splt in ['train', 'valid', 'test']
        } for lang in params.langs
    }
    for paths in params.xlm_classifier_train_dataset.values():
        for p1, p2 in paths.values():
            assert os.path.isfile(p1), "%s not found" % p1
            assert os.path.isfile(p2), "%s not found" % p2

def load_tokenized_data(path, params, text_transform):
    """
    Load tokenized data and returns a list of sentences.
    """
    with open(path, "r", encoding='utf-8') as f:
        data = [text_transform(line.rstrip()) for line in f]
    return data

def load_tst_train_data(params, logger, text_transform):
    data = {}
    data['xlm_classifier'] = {}
    
    for lang in params.langs:
        for splt in ['train', 'valid', 'test']:
            style_data = []
            labels = []
            for label in params.labels:
                style_data += load_tokenized_data(params.xlm_classifier_train_dataset[lang][splt][label], params, text_transform)
                labels += [label] * len(style_data)
            data['xlm_classifier'][splt] = SentenceDataset(style_data, labels)

    # TST train data summary
    logger.info('============ Data summary')
    for data_set, v in data['xlm_classifier'].items():
        logger.info('{: <18} - {: >5} - {: >10}'.format('XLM Classifier train data', data_set, len(v)))

    return data

def collate_fn(batch):
    """
    Collate function to be used when creating a DataLoader.
    """
    data = [item[0] for item in batch] # Size B X S
    labels = [item[1] for item in batch]
    return data, labels

def main(params):

    # initialize the experiment
    logger = initialize_exp(params)
    
    if not os.path.isfile(params.output_path):
        params.output_path = os.path.join(params.dump_path, "%s-%s.txt" % (params.src_lang, params.tgt_lang))

    padding_idx = 1
    bos_idx = 0
    eos_idx = 2
    DEVICE = 'cuda'

    num_classes = 2
    input_dim = 768
    classifier_head = RobertaClassificationHead(num_classes=num_classes, input_dim=input_dim)
    xlm_classifier = XLMR_BASE_ENCODER.get_model(head=classifier_head)
    xlm_classifier.to('cuda')

    text_transform = XLMR_BASE_ENCODER.transform()

    data = load_tst_train_data(params, logger, text_transform)
    learning_rate = 1e-5
    optim = AdamW(xlm_classifier.parameters(), lr=learning_rate)
    criteria = nn.CrossEntropyLoss()

    score_tracker = []
    epoch_for_best_score = -1
    for epoch in range(params.max_epoch):
        xlm_classifier.train()
        train_loader = DataLoader(data['xlm_classifier']['train'], batch_size = 32, shuffle=True, collate_fn=collate_fn)
        for batch in train_loader:
            input = F.to_tensor(batch[0], padding_value=padding_idx).to(DEVICE)
            output = xlm_classifier(input)
            loss = criteria(output, torch.Tensor(batch[1], dtype=torch.long).to(DEVICE))
            optim.zero_grad()
            loss.backward()
            optim.step()

        with torch.no_grad():
            xlm_classifier.eval()
            valid_loader = DataLoader(data['xlm_classifier']['valid'], batch_size = 32, shuffle=False, collate_fn=collate_fn)
            total = 0
            correct = 0
            total_loss = 0
            for batch in valid_loader:
                input = F.to_tensor(batch[0], padding_value=padding_idx).to(DEVICE)
                output = xlm_classifier(input)
                loss = criteria(output, torch.Tensor(batch[1], dtype=torch.long).to(DEVICE))
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += len(batch[1])
                correct += (predicted == batch[1]).sum().item()
            accuracy = correct / total
            logger.info(f'Epoch {epoch} - Loss: {total_loss} - Accuracy: {accuracy}')
            if epoch_for_best_score == -1 or accuracy > max(score_tracker):
                epoch_for_best_score = epoch
                torch.save(xlm_classifier.state_dict(), "best_model.pt")
                print(f"Best model saved for epoch {epoch}.")
            score_tracker.append(accuracy)
            if epoch - epoch_for_best_score > 5:
                break
        torch.save(xlm_classifier.state_dict(), f"model_{epoch}.pt")

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    assert params.src_lang != '' and params.tgt_lang != '' # and params.src_lang != params.tgt_lang
    # assert params.output_path and not os.path.isfile(params.output_path)

    check_params(params)
    main(params)
