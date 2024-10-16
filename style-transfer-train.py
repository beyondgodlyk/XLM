import json
import os
import argparse
import torch
import random

from xlm.utils import AttrDict
from xlm.utils import bool_flag, initialize_exp
from xlm.data.loader import load_binarized, set_dico_parameters
from xlm.data.dictionary import Dictionary
from xlm.model.transformer import TransformerModel
from tst.tst_dataset import TSTDataset
from tst.tst_trainer import TSTTrainer
from tst.tst_evaluator import TSTEvaluator
from tst.classifier import Classifier


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Train classifier used for ST using the DAE trained on MT")

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
    
    # model / output paths
    parser.add_argument("--dae_model_path", type=str, default="", help="DAE Model path")
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
    parser.add_argument("--kernel_sizes", type=str, default="2,3,4,5",
                        help="Kernel sizes for the CNN classifier")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout for the CNN classifier")
    parser.add_argument("--fc_sizes", type=str, default="",
                        help="Fully connected layer sizes for the CNN classifier")
    parser.add_argument("--num_filters", type=int, default=256,
                        help="Number of filters for the CNN classifier")
        
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

    # Parse kernel sizes
    assert params.kernel_sizes != ''
    params.kernel_sizes = list(map(int, params.kernel_sizes.split(',')))

    # Parse fc sizes
    params.fc_sizes = list(map(int, params.fc_sizes.split(','))) if params.fc_sizes != '' else []
    assert len(params.fc_sizes) >= 2
    assert params.fc_sizes[-1] == 1
    assert params.fc_sizes[0] == params.num_filters * len(params.kernel_sizes)

    params.tst_train_dataset = {
        lang: {
            splt: (os.path.join(params.data_path, 'tst.%s.0.%s.pth' % (splt, lang)),
                   os.path.join(params.data_path, 'tst.%s.1.%s.pth' % (splt, lang)))
            for splt in ['train', 'valid', 'test']
        } for lang in params.langs
    }
    for paths in params.tst_train_dataset.values():
        for p1, p2 in paths.values():
            assert os.path.isfile(p1), "%s not found" % p1
            assert os.path.isfile(p2), "%s not found" % p2

def load_tst_train_data(params, logger):
    data = {}
    data['tst'] = {}

    for lang in params.langs:
        for label in params.labels:
            data['tst'][label] = {}
            for splt in ['train', 'valid', 'test']:
                style_data = load_binarized(params.tst_train_dataset[lang][splt][label], params)
                set_dico_parameters(params, data, style_data['dico'])

                dataset = TSTDataset(style_data['sentences'], style_data['positions'], params, label)

                dataset.remove_empty_sentences()
                dataset.remove_long_sentences(params.max_len)

                data['tst'][label][splt] = dataset

    # TST train data summary
    logger.info('============ Data summary')
    for label, v in data['tst'].items():
        for data_set in v.keys():
            logger.info('{: <18} - {: >5} - {: >12}:{: >10}'.format('TST train data', data_set, label, len(v[data_set])))

    return data

def reload_models(params, logger):
    reloaded_dae = torch.load(params.dae_model_path)
    dae_model_params = AttrDict(reloaded_dae['params'])

    # update dictionary parameters
    for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
        setattr(params, name, getattr(dae_model_params, name))

    # build dictionary / build encoder / build decoder / reload weights
    dico = Dictionary(reloaded_dae['dico_id2word'], reloaded_dae['dico_word2id'], reloaded_dae['dico_counts'])
    encoder = TransformerModel(dae_model_params, dico, is_encoder=True, with_output=True).cuda().eval()
    decoder = TransformerModel(dae_model_params, dico, is_encoder=False, with_output=True).cuda().eval()
    encoder.load_state_dict(reloaded_dae['encoder'])
    decoder.load_state_dict(reloaded_dae['decoder'])
    params.src_id = dae_model_params.lang2id[params.src_lang]
    params.tgt_id = dae_model_params.lang2id[params.tgt_lang]

    classifier = Classifier(dae_model_params.emb_dim, 
                            params.kernel_sizes, 
                            params.dropout, 
                            params.fc_sizes, 
                            params.num_filters, 
                            params.max_len).cuda()
    logger.debug("Classifier: {}".format(classifier))
    logger.info("Number of parameters (classifier): %i" % sum([p.numel() for p in classifier.parameters() if p.requires_grad]))

    return dico, encoder, decoder, classifier

def main(params):

    # initialize the experiment
    logger = initialize_exp(params)
    
    if not os.path.isfile(params.output_path):
        params.output_path = os.path.join(params.dump_path, "%s-%s.txt" % (params.src_lang, params.tgt_lang))

    data = load_tst_train_data(params, logger)

    dico, encoder, decoder, classifier = reload_models(params, logger)

    
    trainer = TSTTrainer(classifier, encoder, decoder, data, params)
    evaluator = TSTEvaluator(trainer, data, params)

    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_sentences = 0

        while trainer.n_sentences < trainer.epoch_size:
            for label in random.sample(params.labels, len(params.labels)):
                trainer.classifier_step(label)

            trainer.iter()
        
        logger.info("============ End of epoch %i ============" % trainer.epoch)
        
        # Evaluate on classification metrics
        scores = evaluator.run_all_evals(trainer)

        # print / JSON log
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        if params.is_master:
            logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    assert os.path.isfile(params.dae_model_path)
    assert params.src_lang != '' and params.tgt_lang != '' # and params.src_lang != params.tgt_lang
    # assert params.output_path and not os.path.isfile(params.output_path)

    check_params(params)
    main(params)
