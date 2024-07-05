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

        self.encoder.eval()
    
    def classifier_step(self, lang):
        assert lang in ['en']
        self.classifier.train()

        params = self.params

        lang_id = params.lang2id[lang]
        (x, len) = self.get_batch('tst', lang)
        # (x, len) = self.add_noise(x, len)

        langs = x.clone().fill_(lang_id)
        x, len, langs = to_cuda(x, len, langs)

        enc = self.encoder('fwd', x=x, lengths=len, langs=langs, causal=False)
        enc = enc.transpose(0, 1)

        logger.info("encoded shape: %s" % str(enc.shape))
        return 1
