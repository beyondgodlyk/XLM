import os
import argparse
import torch
import torch.nn.functional as F
from torch import linalg as LA
from torch.nn.utils import clip_grad_norm_

from xlm.utils import AttrDict, restore_segmentation
from xlm.utils import to_cuda
from xlm.utils import bool_flag, initialize_exp
from xlm.data.loader import load_binarized, set_dico_parameters
from xlm.data.dictionary import Dictionary
from xlm.model.transformer import TransformerModel
from xlm.optim import get_optimizer
from xlm.evaluation.evaluator import convert_to_text

from tst.tst_dataset import TSTDataset
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
    parser.add_argument("--learning_rates", type=str, default="0.05,0.07,0.09,0.1,0.12",)
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    
    # threshold for accepting the response
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Threshold for accepting the response")
        
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
    params.langs = list(set([params.src_lang, params.tgt_lang]))
    # assert sorted(params.langs) == params.langs
    # params.id2lang = {k: v for k, v in enumerate(sorted(params.langs))}
    # params.lang2id = {k: v for v, k in params.id2lang.items()}
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

    # Parse learning rates
    params.learning_rates = [float(lr) for lr in params.learning_rates.split(',')]

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
    params.lang2id = dae_model_params.lang2id
    # params.src_id = dae_model_params.lang2id[params.src_lang]
    # params.tgt_id = dae_model_params.lang2id[params.tgt_lang]

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

    dico, encoder, decoder, classifier = reload_models(params)

    data = load_tst_test_data(params, logger)

    # Create a tensor like <EOS> <PAD> <PAD>... <EOS> of size (params.max_len + 2) for 
    # appending to x1 for generating decoder output. This is done because the src_mask 
    # for Enc-Dec Attention is made using len1.max() which raises RuntimeError
    padded_tensor = torch.tensor([params.eos_index] + [params.pad_index] * params.max_len + [params.eos_index]).unsqueeze(1) 

    assert params.batch_size == 1

    for (src_lang, tgt_lang) in set([(params.src_lang, params.tgt_lang), (params.tgt_lang, params.src_lang)]):
        outputs = []
        for label_pair in [(0,1), (1,0)]:
            src_iterator = data['tst'][src_lang][label_pair][0].get_iterator(shuffle=False, 
                                                                        group_by_size=False, 
                                                                        n_sentences=-1)
            tgt_iterator = data['tst'][tgt_lang][label_pair][1].get_iterator(shuffle=False,
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

                langs1 = x1.clone().fill_(params.lang2id[src_lang])
                langs2 = x2.clone().fill_(params.lang2id[tgt_lang])

                x1, len1, langs1, x2, len2, langs2 = to_cuda(x1, len1, langs1, x2, len2, langs2)

                enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
                enc1 = enc1.transpose(0, 1)
                enc1_pred = torch.sigmoid(classifier(enc1).squeeze(1))[0]

                enc2 = encoder('fwd', x=x2, lengths=len2, langs=langs2, causal=False)
                enc2 = enc2.transpose(0, 1)
                enc2_pred = torch.sigmoid(classifier(enc2).squeeze(1))[0]


                logger.info("Orig Pred: %.10e and Gold Pred: %.10e and sum %.10e" % (enc1_pred, enc2_pred, enc1_pred + enc2_pred))
                
                output = None

                for lr in params.learning_rates:
                    if output is not None:
                        break

                    # Clone detached encoder output to be modified iteratively
                    modified_enc1 = enc1.detach().clone()
                    modified_enc1.requires_grad = True

                    assert params.optimizer.find('lr') != -1
                    cur_opt_params = params.optimizer.split('=')
                    cur_opt_params[-1] = str(lr)
                    cur_opt_params = '='.join(cur_opt_params)
                    logger.info("Using optimizer with LR: %s" % cur_opt_params)

                    opt = get_optimizer([modified_enc1], cur_opt_params)
                    it = 0
                    
                    prev = convert_to_text(x1, len1, dico, params)[0]
                    prev_pred = enc1_pred

                    while True:
                        
                        prev_modified_enc1 = modified_enc1.detach().clone()
                        
                        score = classifier(modified_enc1).squeeze(1)
                        pred = torch.sigmoid(score)
                        loss = F.binary_cross_entropy_with_logits(score, torch.Tensor([label_pair[1]]).repeat(score.size()).cuda(), reduction='none')
                        opt.zero_grad()
                        loss[0].backward()

                        if params.clip_grad_norm > 0:
                            clip_grad_norm_([modified_enc1], params.clip_grad_norm)
                        opt.step()

                        idx = params.max_len + 2
                        for i in range(params.max_len + 2):
                            if modified_enc1[0][i].min().item() == 0 and modified_enc1[0][i].max().item() == 0:
                                idx = i
                                break

                        chk_idx = params.max_len + 2
                        for i in reversed(range(params.max_len + 2)):
                            if modified_enc1[0][i].min().item() != 0 or modified_enc1[0][i].max().item() != 0:
                                chk_idx = i
                                break
                        
                        logger.info("idx = %d and chk_idx = %d" % (idx, chk_idx))
                        assert idx-1 == chk_idx

                        # Make sure that padded tensor is unchanged. Check if this is required or is even correct
                        assert torch.all(modified_enc1[1] == enc1[1])
                        
                        logger.info("Iteration %d, Pred: %.10e, Loss: %.10e, Gradient Norm: %.10e, LR: %.4e" % 
                                    (it, pred[0], loss[0].item(), LA.matrix_norm(modified_enc1.grad.data[0]).item(), 
                                    opt.param_groups[0]['lr']))
                        
                        idx_len = len1.clone()
                        idx_len[0] = torch.tensor([idx])
                        generated, lengths = decoder.generate(modified_enc1, idx_len, params.lang2id[tgt_lang], max_len = params.max_len + 2)
                        
                        gen = convert_to_text(generated, lengths, dico, params)[0]
                        logger.info("Modified sentence with idx_len: %s" % gen)
                        
                        # Convert generated[1] to padded tensor. This is done because lengths[1] can change to something else other than params.max_len + 2
                        generated[:,1] = padded_tensor.squeeze(1)
                        lengths[1] = torch.tensor([params.max_len + 2])

                        generated_enc1 = encoder('fwd', x=generated, lengths=lengths, langs=langs1, causal=False)
                        generated_enc1 = generated_enc1.transpose(0, 1)

                        generated_score = classifier(generated_enc1).squeeze(1)
                        generated_pred = torch.sigmoid(generated_score)
                        logger.info("Generated Pred: %.10e" % generated_pred[0])

                        # if gen != prev:
                        #     if (generated_pred[0] > prev_pred) if label_pair[1] == 1 else (generated_pred[0] < prev_pred):
                        #         logger.info("Setting modified_enc1 to generated_enc1")
                        #         prev = gen
                        #         prev_pred = generated_pred[0]

                        #         modified_enc1 = generated_enc1.detach().clone()
                        #         modified_enc1.requires_grad = True
                        #         opt = get_optimizer([modified_enc1], cur_opt_params)
                        #     else:
                        #         logger.info("Generated sentence has lower score. Continuing")

                        # if gen != prev:
                        #     logger.info("Setting modified_enc1 to generated_enc1 nonetheless")
                        #     prev = gen
                        #     prev_pred = generated_pred[0]

                        #     modified_enc1 = generated_enc1.detach().clone()
                        #     modified_enc1.requires_grad = True
                        #     opt = get_optimizer([modified_enc1], cur_opt_params)

                        # if torch.all(prev_modified_enc1[0] == modified_enc1[0]) == True:
                        #     logger.info("Modified encoder output has not changed. Continuing")
                        #     break
                        it += 1
                        logger.info("")

                        # Breaking conditions
                        if generated_pred[0] >= 1 - params.threshold if label_pair[1] == 1 else generated_pred[0] <= params.threshold:
                            output = gen
                            logger.info("Breaking since converged")
                            break

                        if it >= 50:
                            if lr == params.learning_rates[-1] and output is None:
                                output = gen
                                logger.info("Couldn't converge. So setting output to previous best sentence saved")
                            else:
                                logger.info("Max iterations reached. Breaking")
                            break
                
                outputs.append(output)
        logger.info("Saving outputs to file")
        output_path = os.path.join(params.dump_path, "%s-%s.txt" % (src_lang, tgt_lang))
        # export sentences to output file / restore BPE segmentation
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(outputs) + '\n')
        restore_segmentation(output_path)








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
