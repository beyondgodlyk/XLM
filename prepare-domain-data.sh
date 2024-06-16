N_THREADS=16    # number of threads in data preprocessing

# Hardcoded the languages
SRC=en
TGT=fr

MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
PROC_PATH=$DATA_PATH/processed/$SRC-$TGT
YELP_PATH=$PWD/yelp
FOURSQ_PATH=$PWD/foursq

# moses
MOSES=$TOOLS_PATH/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$TOOLS_PATH/fastBPE/fast

# BPE / vocab files
BPE_CODES=$PROC_PATH/codes
FULL_VOCAB=$PROC_PATH/vocab.en-fr

# raw and tokenized files of Yelp and FourSquare
YELP_RAW=$MONO_PATH/$SRC/yelp.$SRC
FOURSQ_RAW=$MONO_PATH/$SRC/foursq.$SRC
YELP_TOK=$YELP_RAW.tok
FOURSQ_TOK=$FOURSQ_RAW.tok

# raw and tokenized files of concatenated data (lines for FR are commented out)
DOMAIN_SRC_RAW=$MONO_PATH/$SRC/domain.$SRC
DOMAIN_SRC_TOK=$MONO_PATH/$SRC/domain.$SRC.tok
# DOMAIN_TGT_RAW=$MONO_PATH/$TGT/domain.$TGT
# DOMAIN_TGT_TOK=$MONO_PATH/$TGT/domain.$TGT.tok

# Monolingual BPE data
DOMAIN_SRC_TRAIN_BPE=$PROC_PATH/domain.train.$SRC
# DOMAIN_TGT_TRAIN_BPE=$PROC_PATH/domain.train.$TGT
DOMAIN_SRC_VALID_BPE=$PROC_PATH/domain.valid.$SRC
DOMAIN_TGT_VALID_BPE=$PROC_PATH/domain.valid.$TGT
DOMAIN_SRC_TEST_BPE=$PROC_PATH/domain.test.$SRC
DOMAIN_TGT_TEST_BPE=$PROC_PATH/domain.test.$TGT

# Parallel tokenized data
DOMAIN_PARA_SRC_VALID_TOK=$FOURSQ_PATH/valid.$SRC.tok
DOMAIN_PARA_TGT_VALID_TOK=$FOURSQ_PATH/valid.$TGT.tok
DOMAIN_PARA_SRC_TEST_TOK=$FOURSQ_PATH/test.$SRC.tok
DOMAIN_PARA_TGT_TEST_TOK=$FOURSQ_PATH/test.$TGT.tok

# Parallel BPE data
DOMAIN_PARA_SRC_VALID_BPE=$PROC_PATH/domain.valid.$SRC-$TGT.$SRC
DOMAIN_PARA_TGT_VALID_BPE=$PROC_PATH/domain.valid.$SRC-$TGT.$TGT
DOMAIN_PARA_SRC_TEST_BPE=$PROC_PATH/domain.test.$SRC-$TGT.$SRC
DOMAIN_PARA_TGT_TEST_BPE=$PROC_PATH/domain.test.$SRC-$TGT.$TGT

echo "Please make sure that you have ran get-data-nmt.sh before running this script."

if ! [[ -f "$DOMAIN_SRC_RAW" ]]; then
  echo "Concatenating Yelp and FourSquare EN data..."
  cat $(ls $YELP_PATH/sentiment*detok $FOURSQ_PATH/train*en) > $DOMAIN_SRC_RAW
fi

# Below lines are commented for FR
# if ! [[ -f "$DOMAIN_TGT_RAW" ]]; then
#   echo "Concatenating Yelp and FourSquare EN data..."
#   cat $(ls $YELP_PATH/sentiment*detok $FOURSQ_PATH/train*en) > $DOMAIN_SRC_RAW
# fi

SRC_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR | $TOKENIZER -l $SRC -no-escape -threads $N_THREADS"
TGT_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $TGT | $REM_NON_PRINT_CHAR | $TOKENIZER -l $TGT -no-escape -threads $N_THREADS"

# tokenize training data
if ! [[ -f "$DOMAIN_SRC_TOK" ]]; then
  echo "Tokenize Domain-$SRC training data..."
  eval "cat $DOMAIN_SRC_RAW | $SRC_PREPROCESSING > $DOMAIN_SRC_TOK"
fi

# apply BPE codes
if ! [[ -f "$DOMAIN_SRC_TRAIN_BPE" ]]; then
  echo "Applying BPE codes to Domain-$SRC..."
  $FASTBPE applybpe $DOMAIN_SRC_TRAIN_BPE $DOMAIN_SRC_TOK $BPE_CODES
fi

# binarize data
if ! [[ -f "$DOMAIN_SRC_TRAIN_BPE.pth" ]]; then
  echo "Binarizing Domain-$SRC training data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $DOMAIN_SRC_TRAIN_BPE
fi

echo "Tokenizing valid and test data..."
eval "cat $FOURSQ_PATH/valid.$SRC | $SRC_PREPROCESSING > $DOMAIN_PARA_SRC_VALID_TOK"
eval "cat $FOURSQ_PATH/valid.$TGT | $TGT_PREPROCESSING > $DOMAIN_PARA_TGT_VALID_TOK"
eval "cat $FOURSQ_PATH/test.$SRC  | $SRC_PREPROCESSING > $DOMAIN_PARA_SRC_TEST_TOK"
eval "cat $FOURSQ_PATH/test.$TGT  | $TGT_PREPROCESSING > $DOMAIN_PARA_TGT_TEST_TOK"

echo "Applying BPE to valid and test files..."
$FASTBPE applybpe $DOMAIN_PARA_SRC_VALID_BPE $DOMAIN_PARA_SRC_VALID_TOK $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $DOMAIN_PARA_TGT_VALID_BPE $DOMAIN_PARA_TGT_VALID_TOK $BPE_CODES $TGT_VOCAB
$FASTBPE applybpe $DOMAIN_PARA_SRC_TEST_BPE  $DOMAIN_PARA_SRC_TEST_TOK  $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $DOMAIN_PARA_TGT_TEST_BPE  $DOMAIN_PARA_TGT_TEST_TOK  $BPE_CODES $TGT_VOCAB

echo "Binarizing data..."
rm -f $DOMAIN_PARA_SRC_VALID_BPE.pth $DOMAIN_PARA_TGT_VALID_BPE.pth $DOMAIN_PARA_SRC_TEST_BPE.pth $DOMAIN_PARA_TGT_TEST_BPE.pth
$MAIN_PATH/preprocess.py $FULL_VOCAB $DOMAIN_PARA_SRC_VALID_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $DOMAIN_PARA_TGT_VALID_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $DOMAIN_PARA_SRC_TEST_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $DOMAIN_PARA_TGT_TEST_BPE

#
# Link monolingual validation and test data to parallel data
#
ln -sf $DOMAIN_PARA_SRC_VALID_BPE.pth $DOMAIN_SRC_VALID_BPE.pth
ln -sf $DOMAIN_PARA_TGT_VALID_BPE.pth $DOMAIN_TGT_VALID_BPE.pth
ln -sf $DOMAIN_PARA_SRC_TEST_BPE.pth  $DOMAIN_SRC_TEST_BPE.pth
ln -sf $DOMAIN_PARA_TGT_TEST_BPE.pth  $DOMAIN_TGT_TEST_BPE.pth


#
# Summary
#
echo ""
echo "===== Data summary"
echo "Monolingual domain training data:"
echo "    $SRC: $DOMAIN_SRC_TRAIN_BPE.pth"
# echo "    $TGT: $DOMAIN_TGT_TRAIN_BPE.pth"
echo "Monolingual domain validation data:"
echo "    $SRC: $DOMAIN_SRC_VALID_BPE.pth"
echo "    $TGT: $DOMAIN_TGT_VALID_BPE.pth"
echo "Monolingual test data:"
echo "    $SRC: $DOMAIN_SRC_TEST_BPE.pth"
echo "    $TGT: $DOMAIN_TGT_TEST_BPE.pth"
echo "Parallel validation data:"
echo "    $SRC: $DOMAIN_PARA_SRC_VALID_BPE.pth"
echo "    $TGT: $DOMAIN_PARA_TGT_VALID_BPE.pth"
echo "Parallel test data:"
echo "    $SRC: $DOMAIN_PARA_SRC_TEST_BPE.pth"
echo "    $TGT: $DOMAIN_PARA_TGT_TEST_BPE.pth"
echo ""
