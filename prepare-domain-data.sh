set -e

N_THREADS=16    # number of threads in data preprocessing

# Hardcoded the languages
SRC=en
TGT=fr

MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
MONO_PATH=$DATA_PATH/mono
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

# raw and tokenized files of different datasets
DOMAIN_MIXED_SRC_RAW=$MONO_PATH/$SRC/domain.mixed.$SRC
DOMAIN_MIXED_SRC_TOK=$MONO_PATH/$SRC/domain.mixed.$SRC.tok
DOMAIN_YELP_SRC_RAW=$MONO_PATH/$SRC/domain.yelp.$SRC
DOMAIN_YELP_SRC_TOK=$MONO_PATH/$SRC/domain.yelp.$SRC.tok
DOMAIN_FOURSQ_SRC_RAW=$MONO_PATH/$SRC/domain.foursq.$SRC
DOMAIN_FOURSQ_SRC_TOK=$MONO_PATH/$SRC/domain.foursq.$SRC.tok
DOMAIN_FOURSQ_TGT_RAW=$MONO_PATH/$TGT/domain.foursq.$TGT
DOMAIN_FOURSQ_TGT_TOK=$MONO_PATH/$TGT/domain.foursq.$TGT.tok

# Monolingual BPE data
DOMAIN_MIXED_SRC_TRAIN_BPE=$PROC_PATH/domain.mixed.train.$SRC
DOMAIN_YELP_SRC_TRAIN_BPE=$PROC_PATH/domain.yelp.train.$SRC
DOMAIN_FOURSQ_SRC_TRAIN_BPE=$PROC_PATH/domain.foursq.train.$SRC
DOMAIN_FOURSQ_TGT_TRAIN_BPE=$PROC_PATH/domain.foursq.train.$TGT
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

echo "Please run get-data-nmt.sh before running this script."

# Below line just checks if the mixed data is already present or not, since it's enough
if ! [[ -f "$DOMAIN_MIXED_SRC_RAW" ]]; then
  echo "Creating Yelp+FourSquare(mixed), Yelp and FourSquare EN data..."
  cat $(ls $YELP_PATH/sentiment*detok $FOURSQ_PATH/train*en) > $DOMAIN_MIXED_SRC_RAW
  cat $(ls $YELP_PATH/sentiment*detok) > $DOMAIN_YELP_SRC_RAW
  cat $(ls $FOURSQ_PATH/train*en) > $DOMAIN_FOURSQ_SRC_RAW
fi

# Below line just checks if the FourSquare data is already present or not, since it's enough
if ! [[ -f "$DOMAIN_FOURSQ_TGT_RAW" ]]; then
  echo "Concatenating FourSquare FR data..."
  cat $(ls $FOURSQ_PATH/train*fr) > $DOMAIN_FOURSQ_TGT_RAW
fi

SRC_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR | $TOKENIZER -l $SRC -no-escape -threads $N_THREADS"
TGT_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $TGT | $REM_NON_PRINT_CHAR | $TOKENIZER -l $TGT -no-escape -threads $N_THREADS"

# tokenize training data
if ! [[ -f "$DOMAIN_MIXED_SRC_TOK" ]]; then
  echo "Tokenize EN datasets..."
  eval "cat $DOMAIN_MIXED_SRC_RAW | $SRC_PREPROCESSING > $DOMAIN_MIXED_SRC_TOK"
  eval "cat $DOMAIN_YELP_SRC_RAW | $SRC_PREPROCESSING > $DOMAIN_YELP_SRC_TOK"
  eval "cat $DOMAIN_FOURSQ_SRC_RAW | $SRC_PREPROCESSING > $DOMAIN_FOURSQ_SRC_TOK"
fi
if ! [[ -f "$DOMAIN_FOURSQ_TGT_TOK" ]]; then
  echo "Tokenize FR dataset..."
  eval "cat $DOMAIN_FOURSQ_TGT_RAW | $TGT_PREPROCESSING > $DOMAIN_FOURSQ_TGT_TOK"
fi

# apply BPE codes
if ! [[ -f "$DOMAIN_MIXED_SRC_TRAIN_BPE" ]]; then
  echo "Applying BPE codes to EN tokenized datasets..."
  $FASTBPE applybpe $DOMAIN_MIXED_SRC_TRAIN_BPE $DOMAIN_MIXED_SRC_TOK $BPE_CODES
  $FASTBPE applybpe $DOMAIN_YELP_SRC_TRAIN_BPE $DOMAIN_YELP_SRC_TOK $BPE_CODES
  $FASTBPE applybpe $DOMAIN_FOURSQ_SRC_TRAIN_BPE $DOMAIN_FOURSQ_SRC_TOK $BPE_CODES
fi
if ! [[ -f "$DOMAIN_FOURSQ_TGT_TRAIN_BPE" ]]; then
  echo "Applying BPE codes to FR datasets..."
  $FASTBPE applybpe $DOMAIN_FOURSQ_TGT_TRAIN_BPE $DOMAIN_FOURSQ_TGT_TOK $BPE_CODES
fi

# binarize data
if ! [[ -f "$DOMAIN_MIXED_SRC_TRAIN_BPE.pth" ]]; then
  echo "Binarizing EN training data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $DOMAIN_MIXED_SRC_TRAIN_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $DOMAIN_YELP_SRC_TRAIN_BPE
  $MAIN_PATH/preprocess.py $FULL_VOCAB $DOMAIN_FOURSQ_SRC_TRAIN_BPE
fi
if ! [[ -f "$DOMAIN_FOURSQ_TGT_TRAIN_BPE.pth" ]]; then
  echo "Binarizing FR training data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $DOMAIN_FOURSQ_TGT_TRAIN_BPE
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
echo "    $SRC: $DOMAIN_MIXED_SRC_TRAIN_BPE.pth"
echo "    $SRC: $DOMAIN_YELP_SRC_TRAIN_BPE.pth"
echo "    $SRC: $DOMAIN_FOURSQ_SRC_TRAIN_BPE.pth"
echo "    $TGT: $DOMAIN_FOURSQ_TGT_TRAIN_BPE.pth"
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

: '
===== Data summary
Monolingual domain training data:
    en: /home/hiwi/rohan/Thesis/XLM/data/processed/en-fr/domain.mixed.train.en.pth
    en: /home/hiwi/rohan/Thesis/XLM/data/processed/en-fr/domain.yelp.train.en.pth
    en: /home/hiwi/rohan/Thesis/XLM/data/processed/en-fr/domain.foursq.train.en.pth
    fr: /home/hiwi/rohan/Thesis/XLM/data/processed/en-fr/domain.foursq.train.fr.pth
Monolingual domain validation data:
    en: /home/hiwi/rohan/Thesis/XLM/data/processed/en-fr/domain.valid.en.pth
    fr: /home/hiwi/rohan/Thesis/XLM/data/processed/en-fr/domain.valid.fr.pth
Monolingual test data:
    en: /home/hiwi/rohan/Thesis/XLM/data/processed/en-fr/domain.test.en.pth
    fr: /home/hiwi/rohan/Thesis/XLM/data/processed/en-fr/domain.test.fr.pth
Parallel validation data:
    en: /home/hiwi/rohan/Thesis/XLM/data/processed/en-fr/domain.valid.en-fr.en.pth
    fr: /home/hiwi/rohan/Thesis/XLM/data/processed/en-fr/domain.valid.en-fr.fr.pth
Parallel test data:
    en: /home/hiwi/rohan/Thesis/XLM/data/processed/en-fr/domain.test.en-fr.en.pth
    fr: /home/hiwi/rohan/Thesis/XLM/data/processed/en-fr/domain.test.en-fr.fr.pth
'