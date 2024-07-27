import os
import argparse
import torch
import torch.nn.functional as F
from torch import linalg as LA
from torch.nn.utils import clip_grad_norm_
import sacrebleu

from xlm.utils import AttrDict
from xlm.utils import to_cuda
from xlm.utils import bool_flag, initialize_exp
from xlm.data.loader import load_binarized, set_dico_parameters
from xlm.data.dataset import ParallelDataset
from xlm.data.dictionary import Dictionary
from xlm.model.transformer import TransformerModel
from xlm.optim import get_optimizer
from xlm.evaluation.evaluator import convert_to_text

from tst.tst_dataset import TSTDataset
from tst.tst_trainer import TSTTrainer
from tst.tst_evaluator import TSTEvaluator
from tst.classifier import Classifier

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Use Classifier for TST")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    # batch parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences per batch")
    parser.add_argument("--tokens_per_batch", type=int, default=-1,
                        help="Number of tokens per batch")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
    
    # model parameters
    parser.add_argument("--gelu_activation", type=bool_flag, default=True,
                        help="Use a GELU activation instead of ReLU")
    
    # model / output paths
    parser.add_argument("--dae_model_path", type=str, default="", help="DAE Model path")
    parser.add_argument("--classifier_model_path", type=str, default="", help="Classifier Model path")
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

    # training parameters
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
        
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
    parser.add_argument("--remove_long_sentences", type=bool_flag, default=False,
                        help="True if long sentences are removed from the dataset")
    
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

    # check datasets
    required_tst = set(params.langs)
    params.tst_test_dataset = {
        lang: {
            label_pairs: (os.path.join(params.data_path, 'tst.%s.test.%s-%s.%s.pth' % (lang, label_pairs[0], label_pairs[1], label_pairs[0])),
                   os.path.join(params.data_path, 'tst.%s.test.%s-%s.%s.pth' % (lang, label_pairs[0], label_pairs[1], label_pairs[1])))
            for label_pairs in [(0, 1), (1, 0)]
        } for lang in params.langs if lang in required_tst
    }
    for paths in params.tst_test_dataset.values():
        for p1, p2 in paths.values():
            assert os.path.isfile(p1), "%s not found" % p1
            assert os.path.isfile(p2), "%s not found" % p2

def load_tst_test_data(params, logger):
    data = {}
    data['tst'] = {}

    for lang in params.langs:
        data['tst'][lang] = {}
        for label_pair in [(0, 1), (1, 0)]: # (0, 1) for positive to negative, (1, 0) for negative to positive
            data['tst'][lang][label_pair] = {}
            src_style_data_path, tgt_style_data_path = params.tst_test_dataset[lang][label_pair]

            src_style_data = load_binarized(src_style_data_path, params)
            tgt_style_data = load_binarized(tgt_style_data_path, params)

            set_dico_parameters(params, data, src_style_data['dico'])
            set_dico_parameters(params, data, tgt_style_data['dico'])

            src_tst_dataset = TSTDataset(src_style_data['sentences'], src_style_data['positions'], params, label_pair[0])
            tgt_tst_dataset = TSTDataset(tgt_style_data['sentences'], tgt_style_data['positions'], params, label_pair[1])
            
            if params.remove_long_sentences:
                src_tst_dataset.remove_long_sentences(params.max_len)
                tgt_tst_dataset.remove_long_sentences(params.max_len)

            data['tst'][lang][label_pair] = (src_tst_dataset, tgt_tst_dataset)

    # TST test data summary
    logger.info('============ Data summary')
    for lang, v in data['tst'].items():
        for label_pair in v.keys():
            logger.info('{: <18} - {: >5} - {: >12}:{: >10}'.format('TST test data', lang, '%s-%s' % label_pair, len(v[label_pair][0])))

    return data

def reload_models(params):
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

    # build classifier
    reloaded_classifier = torch.load(params.classifier_model_path)
    classifier_model_params = AttrDict(reloaded_classifier['params'])
    classifier = Classifier(dae_model_params.emb_dim, 
                            classifier_model_params.kernel_sizes, 
                            classifier_model_params.dropout, 
                            classifier_model_params.fc_sizes, 
                            classifier_model_params.num_filters, 
                            classifier_model_params.max_len).cuda().eval()
    classifier.load_state_dict(reloaded_classifier['classifier'])
    params.max_len = classifier_model_params.max_len

    return dico, encoder, decoder, classifier

# def get_transferred_sentence(len1, lang2_id, enc, decoder, dico, params):
#     generated, lengths = decoder.generate(enc, len1, lang2_id, max_len = params.max_len + 2)
#     return convert_to_text(generated, lengths, dico, params)

def main(params):

    # initialize the experiment
    logger = initialize_exp(params)
    
    if not os.path.isfile(params.output_path):
        params.output_path = os.path.join(params.dump_path, "%s-%s.txt" % (params.src_lang, params.tgt_lang))

    dico, encoder, decoder, classifier = reload_models(params)

    data = load_tst_test_data(params, logger)

    # Create a tensor like <EOS> <PAD> <PAD>... <EOS> of size (params.max_len + 2) for 
    # appending to x1 for generating decoder output. This is done because the src_mask 
    # for Enc-Dec Attention is made using len1.max() which raises RuntimeError
    padded_tensor = torch.tensor([params.eos_index] + [params.pad_index] * params.max_len + [params.eos_index]).unsqueeze(1) 

    assert params.batch_size == 1

    for lang in params.langs:
        for label_pair in [(0,1), (1,0)]:
            src_iterator = data['tst'][lang][label_pair][0].get_iterator(shuffle=False, 
                                                                        group_by_size=False, 
                                                                        n_sentences=-1)
            tgt_iterator = data['tst'][lang][label_pair][1].get_iterator(shuffle=False,
                                                                        group_by_size=False,
                                                                        n_sentences=-1)
            
            for src_batch, tgt_batch in zip(src_iterator, tgt_iterator):
                (x1, len1) = src_batch # x1.size() is (params.max_len + 2, bs), len1.size() is (bs)
                (x2, len2) = tgt_batch

                x1 = torch.cat((x1, padded_tensor), dim=1)
                len1 = torch.cat((len1, torch.tensor([params.max_len + 2])), dim=0)

                x2 = torch.cat((x2, padded_tensor), dim=1)
                len2 = torch.cat((len2, torch.tensor([params.max_len + 2])), dim=0)
                logger.info("=================================================================")
                logger.info("Original sentence: %s" % convert_to_text(x1, len1, dico, params)[0])
                logger.info("Gold sentence: %s" % convert_to_text(x2, len2, dico, params)[0])
                logger.info("")

                langs1 = x1.clone().fill_(params.src_id)
                langs2 = x2.clone().fill_(params.tgt_id)

                x1, len1, langs1, x2, len2, langs2 = to_cuda(x1, len1, langs1, x2, len2, langs2)

                enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
                enc1 = enc1.transpose(0, 1)

                enc2 = encoder('fwd', x=x2, lengths=len2, langs=langs2, causal=False)
                enc2 = enc2.transpose(0, 1)
                enc2_pred = torch.sigmoid(classifier(enc2).squeeze(1))[0]
                logger.info("Gold Pred: %.10e" % enc2_pred)

                # Clone detached encoder output to be modified iteratively
                modified_enc1 = enc1.detach().clone()
                modified_enc1.requires_grad = True

                modified_len1 = len1.clone()
                
                opt = get_optimizer([modified_enc1], params.optimizer)
                it = 0
                
                while True:
                    
                    logger.info("L2 dist b/w orig, modi and modi, gold enc output: %.4e, %.4e" %
                                (LA.vector_norm(torch.mean(enc1[0], dim=0) - torch.mean(modified_enc1[0], dim=0)).item(), 
                                 LA.vector_norm(torch.mean(modified_enc1[0], dim=0) - torch.mean(enc2[0], dim=0)).item()))
                    logger.info("Cosine distance b/w orig, modi and modi, gold enc output: %.4e, %.4e" %
                                (1 - F.cosine_similarity(torch.mean(enc1[0], dim=0).unsqueeze(0), torch.mean(modified_enc1[0], dim=0).unsqueeze(0)).item(),
                                 1 - F.cosine_similarity(torch.mean(modified_enc1[0], dim=0).unsqueeze(0), torch.mean(enc2[0], dim=0).unsqueeze(0)).item()))

                    prev_modified_enc1 = modified_enc1.detach().clone()
                    
                    score = classifier(modified_enc1).squeeze(1)
                    pred = torch.sigmoid(score)
                    loss = F.binary_cross_entropy_with_logits(score, torch.Tensor([label_pair[1]]).repeat(score.size()).cuda(), reduction='none')

                    opt.zero_grad()

                    loss[0].backward()

                    # Set gradients after len1[0] to be zero. Actually wrong because length changes. 
                    # This makes sure that the output sentence can never be greater than len1[0] because 
                    # the gradient just keeps on becoming 0. Although using modified_len1[0] is a correct.
                    # Good way to restrict though.
                    # modified_enc1.grad[0][len1[0]:] = 0
                    # modified_enc1.grad[0][modified_len1[0]:] = 0

                    if params.clip_grad_norm > 0:
                        clip_grad_norm_([modified_enc1], params.clip_grad_norm)
                    opt.step()

                    # Print norms of gradients for each token. Used to verify if the gradient is focused on style tokens
                    # logger.info([(i, LA.vector_norm(modified_enc1.grad[0][i]).item()) for i in range(params.max_len + 2)])
                    
                    # Print min and max of each token
                    # logger.info([(i, modified_enc1[0][i].min().item(), modified_enc1[0][i].max().item()) for i in range(params.max_len + 2)])
                    idx = params.max_len + 2
                    for i in range(params.max_len + 2):
                        if modified_enc1[0][i].min().item() == 0 and modified_enc1[0][i].max().item() == 0:
                            idx = i
                            break
                    logger.info("First token with zero: %d" % idx)
                    
                    # Make sure that padded tensor is unchanged. Check if this is required or is even correct
                    assert torch.all(modified_enc1[1] == enc1[1])
                    
                    # logger.info("Min and max of Gradient: %.4e, %.4e" % (modified_enc1.grad.data[0].min().item(),
                                                                            # modified_enc1.grad.data[0].max().item()))
                    logger.info("Iteration %d, Pred: %.10e, Loss: %.10e, Gradient Norm: %.10e, LR: %.4e" % 
                                (it, pred[0], loss[0].item(), LA.matrix_norm(modified_enc1.grad.data[0]).item(), 
                                 opt.param_groups[0]['lr']))
                    
                    # Decide whether to use modified_len1 or len1 or idx
                    idx_len = len1.clone()
                    idx_len[0] = torch.tensor([idx])
                    generated, lengths = decoder.generate(modified_enc1, idx_len, params.tgt_id, max_len = params.max_len + 2)
                    # generated, lengths = decoder.generate(modified_enc1, modified_len1, params.tgt_id, max_len = params.max_len + 2)
                    
                    logger.info("Modified sentence with len1: %s" % convert_to_text(generated, lengths, dico, params)[0])
                    
                    # Change length corresponding to modified_enc1[0] to be the length of generated[0]
                    modified_len1[0] = lengths[0]

                    # Convert generated[1] to padded tensor. This is done because lengths[1] can change to something else other than params.max_len + 2
                    generated[:,1] = padded_tensor.squeeze(1)
                    lengths[1] = torch.tensor([params.max_len + 2])

                    generated_enc1 = encoder('fwd', x=generated, lengths=lengths, langs=langs1, causal=False)
                    generated_enc1 = generated_enc1.transpose(0, 1)

                    generated_score = classifier(generated_enc1).squeeze(1)
                    generated_pred = torch.sigmoid(generated_score)
                    logger.info("Generated Pred: %.10e" % generated_pred[0])

                    # orig and gold are considered references and gen is hypothesis
                    gen = convert_to_text(generated, lengths, dico, params)[0]
                    orig = convert_to_text(x1, len1, dico, params)[0]
                    gold = convert_to_text(x2, len2, dico, params)[0]
                    logger.info("BLEU score between gen,orig and gen,gold: %.4f, %.4f" %
                                (sacrebleu.corpus_bleu([gen], [[orig]], tokenize='none').score,
                                 sacrebleu.corpus_bleu([gen], [[gold]], tokenize='none').score))
                    logger.info("L2 dist b/w orig, gen and gen, gold enc output: %.4e, %.4e" %
                                (LA.vector_norm(torch.mean(enc1[0], dim=0) - torch.mean(generated_enc1[0], dim=0)).item(), 
                                 LA.vector_norm(torch.mean(generated_enc1[0], dim=0) - torch.mean(enc2[0], dim=0)).item()))
                    logger.info("Cosine distance b/w orig, gen and gen, gold enc output: %.4e, %.4e" %
                                (1 - F.cosine_similarity(torch.mean(enc1[0], dim=0).unsqueeze(0), torch.mean(generated_enc1[0], dim=0).unsqueeze(0)).item(),
                                 1 - F.cosine_similarity(torch.mean(generated_enc1[0], dim=0).unsqueeze(0), torch.mean(enc2[0], dim=0).unsqueeze(0)).item()))
                    # logger.info((1 - generated_pred[0].item() <= torch.sigmoid(classifier(enc1).squeeze(1))[0].item())) 
                    logger.info(modified_len1)

                    if torch.all(prev_modified_enc1[0] == modified_enc1[0]) == True:
                        logger.info("Modified encoder output has not changed. Continuing")
                        break
                    it += 1
                    logger.info("")
                    if it >= 100 or (generated_pred[0] >= 0.99 if label_pair[1] == 1 else generated_pred[0] <= 0.01):
                        break
                # TODO : restore segmentation

                    






if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    assert os.path.isfile(params.dae_model_path)
    assert os.path.isfile(params.classifier_model_path)
    assert params.src_lang != '' and params.tgt_lang != ''
    # assert params.output_path and not os.path.isfile(params.output_path)

    check_params(params)
    main(params)
