set -e

N_THREADS=16    # number of threads in data preprocessing

SRC=en
TGT=fr

MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
MONO_PATH=$DATA_PATH/mono
PROC_PATH=$DATA_PATH/processed/$SRC-$TGT
TST_PROC_PATH=$PROC_PATH/tst
YELP_PATH=$PWD/yelp
TST_PATH=$MONO_PATH/$SRC/tst

# Create folders for TST data
mkdir -p $TST_PATH
mkdir -p $TST_PROC_PATH

# moses
MOSES=$TOOLS_PATH/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
LOWERCASE=$MOSES/scripts/tokenizer/lowercase.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$TOOLS_PATH/fastBPE/fast

# BPE / vocab files
BPE_CODES=$PROC_PATH/codes
FULL_VOCAB=$PROC_PATH/vocab.en-fr

# TST related datasets (train, valid, test) - Note that all datasets are lowercased as well even though it's not in the name
TST_TRAIN_0_TOK=$TST_PATH/tst.train.0.$SRC.tok
TST_TRAIN_1_TOK=$TST_PATH/tst.train.1.$SRC.tok
TST_VALID_0_TOK=$TST_PATH/tst.valid.0.$SRC.tok
TST_VALID_1_TOK=$TST_PATH/tst.valid.1.$SRC.tok
TST_TEST_0_TOK=$TST_PATH/tst.test.0.$SRC.tok
TST_TEST_1_TOK=$TST_PATH/tst.test.1.$SRC.tok

# TST related datasets BPE data
TST_TRAIN_0_BPE=$TST_PROC_PATH/tst.train.0.$SRC
TST_TRAIN_1_BPE=$TST_PROC_PATH/tst.train.1.$SRC
TST_VALID_0_BPE=$TST_PROC_PATH/tst.valid.0.$SRC
TST_VALID_1_BPE=$TST_PROC_PATH/tst.valid.1.$SRC
TST_TEST_0_BPE=$TST_PROC_PATH/tst.test.0.$SRC
TST_TEST_1_BPE=$TST_PROC_PATH/tst.test.1.$SRC

# Also contains script to convert into lowercase
SRC_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR | $TOKENIZER -l $SRC -no-escape -threads $N_THREADS | $LOWERCASE"

if ! [[ -f "$TST_TRAIN_0_TOK" ]]; then
  echo "Creating tokenized TST EN data..."
  eval "cat $YELP_PATH/sentiment.train.0 | $SRC_PREPROCESSING > $TST_TRAIN_0_TOK"
  eval "cat $YELP_PATH/sentiment.train.1 | $SRC_PREPROCESSING > $TST_TRAIN_1_TOK"
  eval "cat $YELP_PATH/sentiment.dev.0 | $SRC_PREPROCESSING > $TST_VALID_0_TOK"
  eval "cat $YELP_PATH/sentiment.dev.1 | $SRC_PREPROCESSING > $TST_VALID_1_TOK"
  eval "cat $YELP_PATH/sentiment.test.0 | $SRC_PREPROCESSING > $TST_TEST_0_TOK"
  eval "cat $YELP_PATH/sentiment.test.1 | $SRC_PREPROCESSING > $TST_TEST_1_TOK"
fi

# apply BPE codes
if ! [[ -f "$TST_TRAIN_0_BPE" ]]; then
  echo "Applying BPE to TST EN data..."
  $FASTBPE applybpe $TST_TRAIN_0_BPE $TST_TRAIN_0_TOK $BPE_CODES
  $FASTBPE applybpe $TST_TRAIN_1_BPE $TST_TRAIN_1_TOK $BPE_CODES
  $FASTBPE applybpe $TST_VALID_0_BPE $TST_VALID_0_TOK $BPE_CODES
  $FASTBPE applybpe $TST_VALID_1_BPE $TST_VALID_1_TOK $BPE_CODES
  $FASTBPE applybpe $TST_TEST_0_BPE $TST_TEST_0_TOK $BPE_CODES
  $FASTBPE applybpe $TST_TEST_1_BPE $TST_TEST_1_TOK $BPE_CODES
fi

# binarize data
if ! [[ -f "$TST_TRAIN_0_BPE.pth" ]]; then
  echo "Binarizing TST EN data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_TRAIN_0_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_TRAIN_1_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_VALID_0_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_VALID_1_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_TEST_0_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_TEST_1_BPE
fi

#
# Summary
#
echo ""
echo "===== Data summary"
echo "Monostyle domain training data:"
echo "    $SRC: $TST_TRAIN_0_BPE.pth"
echo "    $SRC: $TST_TRAIN_1_BPE.pth"
echo "Monostyle domain validation data:"
echo "    $SRC: $TST_VALID_0_BPE.pth"
echo "    $SRC: $TST_VALID_1_BPE.pth"
echo "Monostyle domain test data:"
echo "    $SRC: $TST_TEST_0_BPE.pth"
echo "    $SRC: $TST_TEST_1_BPE.pth"