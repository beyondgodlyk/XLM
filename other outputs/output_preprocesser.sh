set -e

N_THREADS=16    # number of threads in data preprocessing

SRC=en
TGT=fr

# Set MAIN_PATH to parent directory
RAW_OUTPUTS_PATH=$PWD
MAIN_PATH=$(dirname "$PWD")
TOOLS_PATH=$MAIN_PATH/tools
DETOKENIZER_PATH=$MAIN_PATH
TOKENIZED_OUTPUTS_PATH=$MAIN_PATH/outputs

# moses
MOSES=$TOOLS_PATH/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
LOWERCASE=$MOSES/scripts/tokenizer/lowercase.perl

# Also contains script to convert into lowercase
SRC_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR | $TOKENIZER -l $SRC -no-escape -threads $N_THREADS | $LOWERCASE"
TGT_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $TGT | $REM_NON_PRINT_CHAR | $TOKENIZER -l $TGT -no-escape -threads $N_THREADS | $LOWERCASE"

for d in */ ; do
    echo "Processing $d"
    cd $d
    mkdir -p $TOKENIZED_OUTPUTS_PATH/$d
    for f in * ; do
        echo "Processing $f"
        echo $(readlink -f $f)
        echo $TOKENIZED_OUTPUTS_PATH/$d$f.detok
        $DETOKENIZER_PATH/detokenize-files.py "$(readlink -f $f)" "$TOKENIZED_OUTPUTS_PATH/$d$f.detok"
        eval "cat $TOKENIZED_OUTPUTS_PATH/$d$f.detok | $SRC_PREPROCESSING > $TOKENIZED_OUPUTS_PATH/$d$f.tok"
        rm $TOKENIZED_OUTPUTS_PATH/$d$f.detok
    done
    cd ..
done