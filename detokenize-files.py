#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.tokenize.treebank import TreebankWordDetokenizer
import os
import sys

def detokenize_files(src_file_path, dest_file_path):
    detokenizer = TreebankWordDetokenizer()
    filename = os.path.basename(src_file_path)
    detokenized_sentences = []
    with open(src_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            detokenized_sentences.append(detokenizer.detokenize(line.split()))

    with open(dest_file_path, 'w', encoding='utf-8') as f:
        for sentence in detokenized_sentences:
            f.write(sentence + '\n')
    f.close()

if __name__ == '__main__':
    src_file_path = sys.argv[1]
    dest_file_path = sys.argv[2]
    assert os.path.isfile(src_file_path), "Source file path does not exist"
    detokenize_files(src_file_path, dest_file_path)