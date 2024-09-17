from collections import OrderedDict
import time
import torch
import torch.nn.functional as F
from torcheval.metrics.functional import binary_accuracy, binary_f1_score, binary_precision, binary_recall
import numpy as np

from xlm.trainer import Trainer
from xlm.utils import to_cuda

from logging import getLogger

logger = getLogger()

class TSTTrainer(Trainer):
    def __init__(self, classifier, encoder, decoder, data, params):
        """
        Initialize trainer.
        """
        self.MODEL_NAMES = ["classifier"]

        self.classifier = classifier
        self.encoder = encoder
        self.decoder = decoder
        self.data = data
        self.params = params

        # Set is_master to True to save models after every epoch
        self.params.is_master = True

        # epoch / iteration size
        self.epoch_size = params.epoch_size
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        # data iterators
        self.iterators = {}

        # set parameters
        self.set_parameters()

        # set optimizers
        self.set_optimizers()

        # stopping criterion used for early stopping
        if params.stopping_criterion != '':
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == '_':
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sentences = 0
        self.stats = OrderedDict(
            [('processed_s', 0), ('processed_w', 0)] + 
            [('BCE-%s' % label, []) for label in [0, 1]] + 
            [('ACC-%s' % label, []) for label in [0, 1]] + 
            [('PREC-%s' % label, []) for label in [0, 1]] +
            [('RECALL-%s' % label, []) for label in [0, 1]] +
            [('F1-%s' % label, []) for label in [0, 1]]
        )
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

    def get_iterator(self, iter_name, label):
        """
        Create a new iterator for a dataset.
        """
        logger.info("Creating new training data iterator (%s) ..." % ','.join([str(x) for x in [iter_name, label] if x is not None]))

        iterator = self.data[iter_name][label]['train'].get_iterator(
            shuffle=True, 
            group_by_size=False, 
            n_sentences=-1)
    
        self.iterators[(iter_name, label)] = iterator
        return iterator
    
    def get_batch(self, iter_name, label):
        assert iter_name in ['tst']
        assert label in [0, 1]

        iterator = self.iterators.get((iter_name, label))
        if iterator is None:
            iterator = self.get_iterator(iter_name, label)
        try:
            x = next(iterator)
        except StopIteration:
            iterator = self.get_iterator(iter_name, label)
            x = next(iterator)
        return x
    
    def add_noise(self, words, lengths):
        """
        Add noise to the encoder input. Copied from trainer.py to ensure the length is max_len + 2
        """
        words, lengths = self.word_shuffle(words, lengths)
        words, lengths = self.word_dropout(words, lengths)
        words, lengths = self.word_blank(words, lengths)
        return words, lengths
    
    def word_dropout(self, x, l):
        """
        Randomly drop input words. Copied from trainer.py to ensure the length is max_len + 2
        """
        if self.params.word_dropout == 0:
            return x, l
        assert 0 < self.params.word_dropout < 1

        # define words to drop
        eos = self.params.eos_index
        assert (x[0] == eos).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_dropout
        keep[0] = 1  # do not drop the start sentence symbol

        sentences = []
        lengths = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == eos
            words = x[:l[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[j, i]]
            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                new_s.append(words[np.random.randint(1, len(words))])
            new_s.append(eos)
            assert len(new_s) >= 3 and new_s[0] == eos and new_s[-1] == eos
            sentences.append(new_s)
            lengths.append(len(new_s))
        # re-construct input
        l2 = torch.LongTensor(lengths)
        x2 = torch.LongTensor(self.params.max_len + 2, l2.size(0)).fill_(self.params.pad_index)
        for i in range(l2.size(0)):
            x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l2

    def word_blank(self, x, l):
        """
        Randomly blank input words. Copied from trainer.py to ensure the length is max_len + 2
        """
        if self.params.word_blank == 0:
            return x, l
        assert 0 < self.params.word_blank < 1

        # define words to blank
        eos = self.params.eos_index
        assert (x[0] == eos).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_blank
        keep[0] = 1  # do not blank the start sentence symbol

        sentences = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == eos
            words = x[:l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[j, i] else self.params.mask_index for j, w in enumerate(words)]
            new_s.append(eos)
            assert len(new_s) == l[i] and new_s[0] == eos and new_s[-1] == eos
            sentences.append(new_s)
        # re-construct input
        x2 = torch.LongTensor(self.params.max_len + 2, l.size(0)).fill_(self.params.pad_index)
        for i in range(l.size(0)):
            x2[:l[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l
    
    def classifier_step(self, label):
        lang = 'en'
        lang_id = self.params.lang2id[lang]

        self.classifier.train()

        
        (x, len) = self.get_batch('tst', label)
        (x, len) = self.add_noise(x, len)

        langs = x.clone().fill_(lang_id)
        x, len, langs = to_cuda(x, len, langs)

        enc = self.encoder('fwd', x=x, lengths=len, langs=langs, causal=False)
        enc = enc.transpose(0, 1)

        score = self.classifier(enc).squeeze(1)
        logger.info(score)
        pred = torch.sigmoid(score)

        tensor_label = torch.Tensor([label]).repeat(score.size()).cuda()

        loss = F.binary_cross_entropy_with_logits(score, tensor_label)
        self.stats['BCE-%s' % label].append(loss.item())
        self.stats['ACC-%s' % label].append(binary_accuracy(pred, tensor_label).item())
        # self.stats['PREC-%s' % label].append(binary_precision(pred, tensor_label).item())
        # self.stats['RECALL-%s' % label].append(binary_recall(pred_label, tensor_label).item())
        # self.stats['F1-%s' % label].append(binary_f1_score(pred, tensor_label).item())
        
        self.optimize(loss)
        
        self.n_sentences += self.params.batch_size
        self.stats['processed_s'] += len.size(0)
        self.stats['processed_w'] += (len - 1).sum().item()
    
    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()
