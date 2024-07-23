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

# TST datasets (train, valid, test) for Classifier training - Note that all datasets are lowercased as well even though it's not in the name
TST_TRAIN_0_TOK=$TST_PATH/tst.train.0.$SRC.tok
TST_TRAIN_1_TOK=$TST_PATH/tst.train.1.$SRC.tok
TST_VALID_0_TOK=$TST_PATH/tst.valid.0.$SRC.tok
TST_VALID_1_TOK=$TST_PATH/tst.valid.1.$SRC.tok
TST_TEST_0_TOK=$TST_PATH/tst.test.0.$SRC.tok
TST_TEST_1_TOK=$TST_PATH/tst.test.1.$SRC.tok

# TST datasets for evaluation - Note that all datasets are lowercased as well even though it's not in the name
TST_SRC_TEST_0_1_0_TOK=$TST_PATH/tst.$SRC.test.0-1.0.tok
TST_SRC_TEST_0_1_1_TOK=$TST_PATH/tst.$SRC.test.0-1.1.tok
TST_SRC_TEST_1_0_0_TOK=$TST_PATH/tst.$SRC.test.1-0.0.tok
TST_SRC_TEST_1_0_1_TOK=$TST_PATH/tst.$SRC.test.1-0.1.tok

TST_TGT_TEST_0_1_0_TOK=$TST_PATH/tst.$TGT.test.0-1.0.tok
TST_TGT_TEST_0_1_1_TOK=$TST_PATH/tst.$TGT.test.0-1.1.tok
TST_TGT_TEST_1_0_0_TOK=$TST_PATH/tst.$TGT.test.1-0.0.tok
TST_TGT_TEST_1_0_1_TOK=$TST_PATH/tst.$TGT.test.1-0.1.tok


# TST datasets for Classifier training - BPE data
TST_TRAIN_0_BPE=$TST_PROC_PATH/tst.train.0.$SRC
TST_TRAIN_1_BPE=$TST_PROC_PATH/tst.train.1.$SRC
TST_VALID_0_BPE=$TST_PROC_PATH/tst.valid.0.$SRC
TST_VALID_1_BPE=$TST_PROC_PATH/tst.valid.1.$SRC
TST_TEST_0_BPE=$TST_PROC_PATH/tst.test.0.$SRC
TST_TEST_1_BPE=$TST_PROC_PATH/tst.test.1.$SRC

# TST datasets for evaluation - BPE data
TST_SRC_TEST_0_1_0_BPE=$TST_PROC_PATH/tst.$SRC.test.0-1.0
TST_SRC_TEST_0_1_1_BPE=$TST_PROC_PATH/tst.$SRC.test.0-1.1
TST_SRC_TEST_1_0_0_BPE=$TST_PROC_PATH/tst.$SRC.test.1-0.0
TST_SRC_TEST_1_0_1_BPE=$TST_PROC_PATH/tst.$SRC.test.1-0.1

TST_TGT_TEST_0_1_0_BPE=$TST_PROC_PATH/tst.$TGT.test.0-1.0
TST_TGT_TEST_0_1_1_BPE=$TST_PROC_PATH/tst.$TGT.test.0-1.1
TST_TGT_TEST_1_0_0_BPE=$TST_PROC_PATH/tst.$TGT.test.1-0.0
TST_TGT_TEST_1_0_1_BPE=$TST_PROC_PATH/tst.$TGT.test.1-0.1


# Also contains script to convert into lowercase
SRC_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR | $TOKENIZER -l $SRC -no-escape -threads $N_THREADS | $LOWERCASE"
TGT_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $TGT | $REM_NON_PRINT_CHAR | $TOKENIZER -l $TGT -no-escape -threads $N_THREADS | $LOWERCASE"

if ! [[ -f "$TST_TRAIN_0_TOK" ]]; then
  echo "Creating tokenized TST EN data for Classifier training..."
  eval "cat $YELP_PATH/sentiment.train.0.detok | $SRC_PREPROCESSING > $TST_TRAIN_0_TOK"
  eval "cat $YELP_PATH/sentiment.train.1.detok | $SRC_PREPROCESSING > $TST_TRAIN_1_TOK"
  eval "cat $YELP_PATH/sentiment.dev.0.detok | $SRC_PREPROCESSING > $TST_VALID_0_TOK"
  eval "cat $YELP_PATH/sentiment.dev.1.detok | $SRC_PREPROCESSING > $TST_VALID_1_TOK"
  eval "cat $YELP_PATH/sentiment.test.0.detok | $SRC_PREPROCESSING > $TST_TEST_0_TOK"
  eval "cat $YELP_PATH/sentiment.test.1.detok | $SRC_PREPROCESSING > $TST_TEST_1_TOK"
fi

if ! [[ -f "$TST_SRC_TEST_0_1_0_TOK" ]]; then
  echo "Creating tokenized TST EN data for evaluation..."
  eval "cat $YELP_PATH/sentiment.test.0.detok | $SRC_PREPROCESSING > $TST_SRC_TEST_0_1_0_TOK"
  eval "cat $YELP_PATH/human.txt.detok | head -n 500 |$SRC_PREPROCESSING > $TST_SRC_TEST_0_1_1_TOK"

  eval "cat $YELP_PATH/sentiment.test.1.detok | $SRC_PREPROCESSING > $TST_SRC_TEST_1_0_1_TOK"
  eval "cat $YELP_PATH/human.txt.detok | tail -n 500 |$SRC_PREPROCESSING > $TST_SRC_TEST_1_0_0_TOK"
fi

if ! [[ -f "$TST_TGT_TEST_0_1_0_TOK" ]]; then
  echo "Creating tokenized TST FR data for evaluation..."
  eval "cat $YELP_PATH/translated.sentiment.test.0.detok | $TGT_PREPROCESSING > $TST_TGT_TEST_0_1_0_TOK"
  eval "cat $YELP_PATH/translated.human.txt.detok | head -n 500 |$TGT_PREPROCESSING > $TST_TGT_TEST_0_1_1_TOK"

  eval "cat $YELP_PATH/translated.sentiment.test.1.detok | $TGT_PREPROCESSING > $TST_TGT_TEST_1_0_1_TOK"
  eval "cat $YELP_PATH/translated.human.txt.detok | tail -n 500 |$TGT_PREPROCESSING > $TST_TGT_TEST_1_0_0_TOK"
fi

# apply BPE codes
if ! [[ -f "$TST_TRAIN_0_BPE" ]]; then
  echo "Applying BPE to TST EN data for Classifier training..."
  $FASTBPE applybpe $TST_TRAIN_0_BPE $TST_TRAIN_0_TOK $BPE_CODES
  $FASTBPE applybpe $TST_TRAIN_1_BPE $TST_TRAIN_1_TOK $BPE_CODES
  $FASTBPE applybpe $TST_VALID_0_BPE $TST_VALID_0_TOK $BPE_CODES
  $FASTBPE applybpe $TST_VALID_1_BPE $TST_VALID_1_TOK $BPE_CODES
  $FASTBPE applybpe $TST_TEST_0_BPE $TST_TEST_0_TOK $BPE_CODES
  $FASTBPE applybpe $TST_TEST_1_BPE $TST_TEST_1_TOK $BPE_CODES
fi

if ! [[ -f "$TST_SRC_TEST_0_1_0_BPE" ]]; then
  echo "Applying BPE to TST EN data for evaluation..."
  $FASTBPE applybpe $TST_SRC_TEST_0_1_0_BPE $TST_SRC_TEST_0_1_0_TOK $BPE_CODES
  $FASTBPE applybpe $TST_SRC_TEST_0_1_1_BPE $TST_SRC_TEST_0_1_1_TOK $BPE_CODES
  $FASTBPE applybpe $TST_SRC_TEST_1_0_0_BPE $TST_SRC_TEST_1_0_0_TOK $BPE_CODES
  $FASTBPE applybpe $TST_SRC_TEST_1_0_1_BPE $TST_SRC_TEST_1_0_1_TOK $BPE_CODES
fi

if ! [[ -f "$TST_TGT_TEST_0_1_0_BPE" ]]; then
  echo "Applying BPE to TST FR data for evaluation..."
  $FASTBPE applybpe $TST_TGT_TEST_0_1_0_BPE $TST_TGT_TEST_0_1_0_TOK $BPE_CODES
  $FASTBPE applybpe $TST_TGT_TEST_0_1_1_BPE $TST_TGT_TEST_0_1_1_TOK $BPE_CODES
  $FASTBPE applybpe $TST_TGT_TEST_1_0_0_BPE $TST_TGT_TEST_1_0_0_TOK $BPE_CODES
  $FASTBPE applybpe $TST_TGT_TEST_1_0_1_BPE $TST_TGT_TEST_1_0_1_TOK $BPE_CODES
fi


# binarize data
if ! [[ -f "$TST_TRAIN_0_BPE.pth" ]]; then
  echo "Binarizing TST EN data for Classifier training..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_TRAIN_0_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_TRAIN_1_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_VALID_0_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_VALID_1_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_TEST_0_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_TEST_1_BPE
fi

if ! [[ -f "$TST_SRC_TEST_0_1_0_BPE.pth" ]]; then
  echo "Binarizing TST EN data for evaluation..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_SRC_TEST_0_1_0_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_SRC_TEST_0_1_1_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_SRC_TEST_1_0_0_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_SRC_TEST_1_0_1_BPE
fi

if ! [[ -f "$TST_TGT_TEST_0_1_0_BPE.pth" ]]; then
  echo "Binarizing TST FR data for evaluation..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_TGT_TEST_0_1_0_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_TGT_TEST_0_1_1_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_TGT_TEST_1_0_0_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TST_TGT_TEST_1_0_1_BPE
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
echo "Evaluation data:"
echo "    $SRC: $TST_SRC_TEST_0_1_0_BPE.pth"
echo "    $SRC: $TST_SRC_TEST_0_1_1_BPE.pth"
echo "    $SRC: $TST_SRC_TEST_1_0_0_BPE.pth"
echo "    $SRC: $TST_SRC_TEST_1_0_1_BPE.pth"
echo "    $TGT: $TST_TGT_TEST_0_1_0_BPE.pth"
echo "    $TGT: $TST_TGT_TEST_0_1_1_BPE.pth"
echo "    $TGT: $TST_TGT_TEST_1_0_0_BPE.pth"
echo "    $TGT: $TST_TGT_TEST_1_0_1_BPE.pth"
echo ""