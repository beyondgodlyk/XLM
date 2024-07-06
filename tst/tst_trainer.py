from collections import OrderedDict

from xlm.trainer import Trainer
from xlm.utils import to_cuda

from logging import getLogger

logger = getLogger()

class TSTTrainer(Trainer):
    def __init__(self, encoder, classifier, data, params):
        self.MODEL_NAMES = ["classifier"]

        self.encoder = encoder
        self.classifier = classifier
        self.data = data
        self.params = params

        super().__init__(data, params)

        self.stats = OrderedDict(
            [('processed_s', 0), ('processed_w', 0)] + 
            [('AE-%s' % label, []) for label in [0, 1]]
        )
        self.encoder.eval()

    def get_iterator(self, iter_name, label):
        """
        Create a new iterator for a dataset.
        """
        logger.info("Creating new training data iterator (%s) ..." % ','.join([str(x) for x in [iter_name, label] if x is not None]))

        iterator = self.data[iter_name][label]['train'].get_iterator(
            shuffle=False, 
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
    
    def classifier_step(self, label):
        lang = 'en'
        self.classifier.train()

        params = self.params

        lang_id = params.lang2id[lang]
        (x, len) = self.get_batch('tst', label)
        # (x, len) = self.add_noise(x, len)

        langs = x.clone().fill_(lang_id)
        x, len, langs = to_cuda(x, len, langs)

        enc = self.encoder('fwd', x=x, lengths=len, langs=langs, causal=False)
        enc = enc.transpose(0, 1)

        logger.info("encoded shape: %s" % str(enc.shape))
        logger.info("encoded: %s" % str(enc))

        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len.size(0)
        self.stats['processed_w'] += (len - 1).sum().item()
        return 1
