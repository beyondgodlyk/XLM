import torch
import numpy as np
import os
src_data = torch.load('./data/processed/en-fr/domain.test.en-fr.en.pth')
tgt_data = torch.load('./data/processed/en-fr/domain.test.en-fr.fr.pth')

dico = src_data['dico']
sent1 = src_data['sentences']
sent2 = tgt_data['sentences']
pos1 = src_data['positions']
pos2 = tgt_data['positions']

lengths1 = pos1[:, 1] - pos1[:, 0]
lengths2 = pos2[:, 1] - pos2[:, 0]

n_sentences = len(pos1)
lengths = lengths1 + lengths2 + 4
indices = np.arange(n_sentences)

indices = indices[np.argsort(lengths[indices], kind='mergesort')]

with open("deepl_translated.test.en-fr", "r", encoding='utf-8') as f:
    initial_en_fr = [line.rstrip() for line in f]

with open("deepl_translated.test.fr-en", "r", encoding='utf-8') as f:
    initial_fr_en = [line.rstrip() for line in f]

rearranged_ou_en_fr = [initial_en_fr[i] for i in indices]
rearranged_ou_fr_en = [initial_fr_en[i] for i in indices]

with open("rearranged_deepl_translated.test.en-fr", "w", encoding='utf-8') as f:
    for line in rearranged_ou_en_fr:
        f.write(line + "\n")

with open("rearranged_deepl_translated.test.fr-en", "w", encoding='utf-8') as f:
    for line in rearranged_ou_fr_en:
        f.write(line + "\n")

# [[dico[token] for token in sent1[pos1[id][0]:pos1[id][1]]] for id in indices[-5:]]