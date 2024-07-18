from collections import OrderedDict
import os
import time
import torch
import torch.nn.functional as F
from torcheval.metrics.functional import binary_accuracy, binary_f1_score, binary_precision, binary_recall
from torch.nn.utils import clip_grad_norm_

from xlm.trainer import Trainer
from xlm.utils import to_cuda

from logging import getLogger

logger = getLogger()

class TSTTrainer(Trainer):
    def __init__(self, classifier, dae_trainer, data, params):
        """
        Initialize trainer.
        """
        # Encoder is also optimized
        self.MODEL_NAMES = ["classifier"]

        self.classifier = classifier
        self.encoder = dae_trainer.encoder
        self.decoder = dae_trainer.decoder
        self.dae_trainer = dae_trainer
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

    def optimize(self, loss):
        """
        Optimize the Encoder and Classifier using 2 optimizers.
        """
        cl_opt_keys = list(self.optimizers.keys())
        assert len(cl_opt_keys) == 1
        cl_optimizer = self.optimizers[cl_opt_keys[0]]

        dae_opt_keys = list(self.dae_trainer.optimizers.keys())
        assert len(dae_opt_keys) == 1
        enc_optimizer = self.dae_trainer.optimizers[dae_opt_keys[0]]

        # Make sure the params of classifier are correctly loaded in the optimizer
        assert self.parameters[cl_opt_keys[0]] == [p for p in list(self.classifier.parameters()) if p.requires_grad]
        # Make sure all the params of Classifier are being updated
        print([p.grad != None for p in list(self.classifier.parameters()) if p.requires_grad])
        assert sum([p.grad != None for p in list(self.classifier.parameters()) if p.requires_grad]) == len(self.parameters[cl_opt_keys[0]])
        
        # Since LHS contains params for Enc+Dec, make sure that the length of params which have grad (only Enc) are less
        assert len(self.parameters[dae_opt_keys[0]]) > sum([p.grad != None for p in list(self.encoder.parameters()) if p.requires_grad])
        # Make sure all the params of Enc are being updated
        assert len([p for p in list(self.encoder.parameters()) if p.requires_grad]) == sum([p.grad != None for p in list(self.encoder.parameters()) if p.requires_grad]) 
        # Makes sure none of the params of Dec are being updated
        assert sum([p.grad != None for p in list(self.decoder.parameters()) if p.requires_grad]) == 0

        cl_optimizer.zero_grad()
        enc_optimizer.zero_grad()

        loss.backward()

        if self.params.clip_grad_norm > 0:
            clip_grad_norm_(self.parameters[cl_opt_keys], self.params.clip_grad_norm)
            clip_grad_norm_(self.parameters[dae_opt_keys], self.params.clip_grad_norm)
        
        cl_optimizer.step()
        enc_optimizer.step()
    
    def tst_step(self, label, lambda_coeff):
        lang = 'en'
        lang_id = self.params.lang2id[lang]

        self.classifier.train()
        self.encoder.train()
        
        (x, len) = self.get_batch('tst', label)
        # (x, len) = self.add_noise(x, len)

        langs = x.clone().fill_(lang_id)
        x, len, langs = to_cuda(x, len, langs)

        enc = self.encoder('fwd', x=x, lengths=len, langs=langs, causal=False)
        enc = enc.transpose(0, 1)

        score = self.classifier(enc).squeeze(1)
        pred = torch.sigmoid(score)

        tensor_label = torch.Tensor([label]).repeat(score.size()).cuda()

        loss = F.binary_cross_entropy_with_logits(score, tensor_label)
        self.stats['BCE-%s' % label].append(loss.item())
        self.stats['ACC-%s' % label].append(binary_accuracy(pred, tensor_label).item())
        # self.stats['PREC-%s' % label].append(binary_precision(pred, tensor_label).item())
        # self.stats['RECALL-%s' % label].append(binary_recall(pred_label, tensor_label).item())
        # self.stats['F1-%s' % label].append(binary_f1_score(pred, tensor_label).item())
        
        loss = lambda_coeff * loss
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

    def save_checkpoint(self, name, include_optimizers=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return
        # Modified name to separate the saved modely in both the trainers
        name = 'classifier' + name
        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
        }

        for name in self.MODEL_NAMES:
            if name != 'encoder': # encoder is saved separately
                logger.warning(f"Saving {name} parameters ...")
                data[name] = getattr(self, name).state_dict()

        if include_optimizers:
            for name in self.optimizers.keys():
                logger.warning(f"Saving {name} optimizer ...")
                data[f'{name}_optimizer'] = self.optimizers[name].state_dict()

        data['dico_id2word'] = self.data['dico'].id2word
        data['dico_word2id'] = self.data['dico'].word2id
        data['dico_counts'] = self.data['dico'].counts
        data['params'] = {k: v for k, v in self.params.__dict__.items()}

        torch.save(data, path)
