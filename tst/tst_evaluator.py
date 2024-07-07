from xlm.evaluation.evaluator import Evaluator

class TSTEvaluator(Evaluator):
    def __init__(self, trainer, data, params):
        """
        Build TST evaluator.
        """
        super().__init__(trainer, data, params)
        self.classifier = trainer.classifier
    
    def evaluate_classifier(self, data_set, label):
        """
        Evaluate classifier on the given dataset.
        """
        