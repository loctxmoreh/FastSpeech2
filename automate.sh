#!/usr/bin/env bash

# Currently working on HAC machines only, A100 machines are not supported

set -e

DATASET_ROOT="/NAS/common_data"
LJSPEECH_LINK="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
ALIGNMENT_GGDRIVE_ID="1ukb8o-SnqhXCxq7drI3zye3tZdrGvQDA"

if [[ ! -d "${DATASET_ROOT}/LJSpeech-1.1" ]]; then
    curl $LJSPEECH_LINK --output $DATASET_ROOT/ljspeech.tar.bz2 --silent
    tar xf $DATASET_ROOT/ljspeech.tar.bz2 -C $DATASET_ROOT
    rm -f $DATASET_ROOT/ljspeech.tar.bz2
fi

[[ ! -f hifigan/generator_LJSpeech.pth.tar ]] \
    && unzip hifigan/generator_LJSpeech.pth.tar.zip -d hifigan/

if [[ ! -d preprocessed_data/LJSpeech/TextGrid ]]; then
    mkdir -p preprocessed_data/LJSpeech
    [[ ! -x "$(command -v gdown)" ]] && echo "gdown not found" && exit 1
    gdown https://drive.google.com/uc?id=$ALIGNMENT_GGDRIVE_ID \
        --output LJSpeech.zip
    unzip LJSpeech.zip -d preprocessed_data/LJSpeech
    rm -f LJSpeech.zip
fi

corpus_path=$(grep corpus_path: config/LJSpeech/preprocess.yaml | awk '{print $NF}' | tr -d '"')
[[ ! -d $corpus_path ]] && echo "${corpus_path} in preprocess.yaml not exist" && exit 1

env_file="hacenv.yml"
env_name=$(grep ^name: $env_file | awk '{print $NF}')

conda env create -f $env_file
conda run -n $env_name update-moreh --force

echo "Running prepare_align.py"
conda run -n $env_name python3 prepare_align.py config/LJSpeech/preprocess.yaml

echo "Running preprocess.py"
conda run -n $env_name python3 preprocess.py config/LJSpeech/preprocess.yaml

train_steps=$(grep total_step: config/LJSpeech/train.yaml | awk '{print $NF}')
[[ $train_step -gt 10000 ]] && echo "Total train steps ${train_steps} too big" && exit 1

echo "Running train.py"
conda run -n $env_name python3 train.py \
    -p config/LJSpeech/preprocess.yaml \
    -m config/LJSpeech/model.yaml \
    -t config/LJSpeech/train.yaml

conda env remove -n $env_name
