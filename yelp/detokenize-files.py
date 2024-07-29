from nltk.tokenize.treebank import TreebankWordDetokenizer

def detokenize_files():
    detokenizer = TreebankWordDetokenizer()
    train_files = ['sentiment.train.0', 'sentiment.train.1', 'sentiment.dev.0', 'sentiment.dev.1', 'sentiment.test.0', 'sentiment.test.1', 'human.txt', 'human-yelp.txt']
    for file in train_files:
        detokenized_sentences = []
        with open(file, 'r') as f:
            for line in f:
                detokenized_sentences.append(detokenizer.detokenize(line.split()))

        with open(file + '.detok', 'w') as f:
            for sentence in detokenized_sentences:
                f.write(sentence + '\n')
        f.close()

if __name__ == '__main__':
    detokenize_files()