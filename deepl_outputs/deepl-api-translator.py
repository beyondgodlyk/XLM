import argparse
import os
import deepl

def get_parser():
    parser = argparse.ArgumentParser(description="Translate text using the DeepL API")
    parser.add_argument("--file_path", type=str, help="Text to translate")
    parser.add_argument("--src-lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt-lang", type=str, default="", help="Target language")
    return parser

def check_params(args):
    print("Checking parameters...")
    assert os.path.isfile(args.file_path), "Please provide a valid file path"
    assert args.src_lang in ["EN", "FR"], "Source language not supported"
    assert args.tgt_lang in ["EN-US", "FR"], "Target language not supported"
    assert args.src_lang != args.tgt_lang, "Source and target languages must be different"
    print("Parameters are valid")

auth_key = "2b4ab54c-8820-4b7f-a8dc-c01d34d5622c:fx"

if __name__ == "__main__":
    batch_size = 50
    parser = get_parser()
    args = parser.parse_args()
    check_params(args)

    with open(args.file_path, "r", encoding='utf-8') as f:
        reviews = [line.rstrip() for line in f]

    batches = [reviews[i:i+batch_size] for i in range(0, len(reviews), batch_size)]
    
    print("Translating text...")
    translator = deepl.Translator(auth_key)
    translated_batches = [translator.translate_text(batch, source_lang=args.src_lang, target_lang=args.tgt_lang, formality="prefer_less", split_sentences=0, preserve_formatting=True) for batch in batches]
    print("Translation has finished")
    
    translated_reviews = [review for batch in translated_batches for review in batch]

    with open("translated." + os.path.basename(args.file_path), "w", encoding='utf-8') as f:
        for review in translated_reviews:
            f.write(review.text + "\n")