import kenlm
import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser(description="Find PPL of a given file")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--file_path", type=str, default="", help="Path to the file")
    parser.add_argument("--print_list", type=bool, default=False, help="Print list of PPLs")
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    assert os.path.exists(args.model_path), "Model path does not exist"
    assert os.path.exists(args.file_path), "File path does not exist"

    model = kenlm.Model(args.model_path)
    with open(args.file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        ppl = []
        sum_score = 0
        tot_words = 0
        for line in data:
            sum_score += model.score(line)
            tot_words += len(line.split())
            ppl.append(model.perplexity(line))
        if args.print_list:
            print('\n'.join([str(p) for p in ppl]))
        print("PPL from score: ", 10**(-sum_score/tot_words))
        print("PPL: ", sum(ppl)/len(ppl))

