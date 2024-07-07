import torch
import torch.nn.functional as F
from torcheval.metrics.functional import binary_accuracy, binary_f1_score, binary_precision, binary_recall

from collections import OrderedDict
from xlm.evaluation.evaluator import Evaluator

from xlm.utils import to_cuda

from logging import getLogger

logger = getLogger()

class TSTEvaluator(Evaluator):
    def __init__(self, trainer, data, params):
        """
        Build TST evaluator.
        """
        self.trainer = trainer
        self.data = data
        self.dico = data['dico']
        self.params = params
        self.encoder = trainer.encoder
        self.classifier = trainer.classifier

    def run_all_evals(self, trainer):
        """
        Run all evaluations.
        """
        scores = OrderedDict({'epoch': trainer.epoch})

        with torch.no_grad():
            for data_set in ['valid', 'test']:
                self.evaluate_classifier(scores, data_set)
        
        return scores

    def get_iterator(self, data_set, label):
        iterator = self.data['tst'][label][data_set].get_iterator(
            shuffle=False, 
            group_by_size=False, 
            n_sentences=-1)
    
        for batch in iterator:
            yield batch
    
    def evaluate_classifier(self, scores, data_set):
        """
        Evaluate classifier on the sentiment data for either valid or test.
        """
        lang = 'en'
        lang_id = self.params.lang2id[lang]

        assert data_set in ['valid', 'test']

        self.classifier.eval()

        agg_pred = torch.Tensor().cuda()
        agg_label = torch.Tensor().cuda()

        for label in self.params.labels:
            for batch in self.get_iterator(data_set, label):
                (x, len) = batch
                
                langs = x.clone().fill_(lang_id)
                x, len, langs = to_cuda(x, len, langs)

                enc = self.encoder('fwd', x=x, lengths=len, langs=langs, causal=False)
                enc = enc.transpose(0, 1)

                pred = self.classifier(enc).squeeze(1)

                agg_pred = torch.cat((agg_pred, pred))
                agg_label = torch.cat((agg_label, torch.Tensor([label]).repeat(pred.size()).cuda()))

        assert agg_pred.size(0) == (4000 if data_set == 'valid' else 1000)
        
        agg_label = agg_label.long() # convert to long because binary_recall() does and operation and gives error

        # Accuracy and BCE for the separate datasets
        scores['BCE-%s-%s' % (data_set, 0)] = F.binary_cross_entropy(agg_pred[:(2000 if data_set == 'valid' else 500)], agg_label[:(2000 if data_set == 'valid' else 500)]).item()
        scores['BCE-%s-%s' % (data_set, 1)] = F.binary_cross_entropy(agg_pred[(2000 if data_set == 'valid' else 500):], agg_label[(2000 if data_set == 'valid' else 500):]).item()
        scores['ACC-%s-%s' % (data_set, 0)] = binary_accuracy(agg_pred[:(2000 if data_set == 'valid' else 500)], agg_label[:(2000 if data_set == 'valid' else 500)]).item()
        scores['ACC-%s-%s' % (data_set, 1)] = binary_accuracy(agg_pred[(2000 if data_set == 'valid' else 500):], agg_label[(2000 if data_set == 'valid' else 500):]).item()
        
        # compute accuracy, precision, recall, f1 of the combined dataset
        scores['ACC-%s' % data_set] = binary_accuracy(agg_pred, agg_label).item()
        scores['PREC-%s' % data_set] = binary_precision(agg_pred, agg_label).item()
        scores['RECALL-%s' % data_set] = binary_recall(agg_pred, agg_label).item()
        scores['F1-%s' % data_set] = binary_f1_score(agg_pred, agg_label).item()
