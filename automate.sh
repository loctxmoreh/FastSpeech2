#!/usr/bin/env bash

set -e

DATASET_ROOT="/data/work/datasets"
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
    gdown https://drive.google.com/uc?id=$ALIGNMENT_GGDRIVE_ID \
        --output LJSpeech.zip
    unzip LJSpeech.zip -d preprocessed_data/LJSpeech
    rm -f LJSpeech.zip
fi

env_file="hacenv.yml"
env_name=$(grep ^name: $env_file | awk '{print $NF}')

conda env create -f $env_file
# conda env run -n $env_name update-moreh --force

conda run -n $env_name python3 prepare_align.py config/LJSpeech/preprocess.yaml
conda run -n $env_name python3 preprocess.py config/LJSpeech/preprocess.yaml
conda run -n $env_name python3 train.py \
    -p config/LJSpeech/preprocess.yaml \
    -m config/LJSpeech/model.yaml \
    -t config/LJSpeech/train.yaml

conda env remove -n $env_name
