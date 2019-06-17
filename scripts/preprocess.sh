#!/usr/bin/env bash

PROJ=/research/king3/yfgao/pd/conversational-QG
PYT=/research/king2/yfgao/envs/pyt0.4.1py3.6cuda9/bin/python
DATA=/research/king3/yfgao/pd/conversational-QG/data

mkdir -p ${DATA}/processed
mkdir -p ${DATA}/model
mkdir -p ${DATA}/pred

${PYT} code/preprocess.py \
        -train_dir=${DATA}/coqg-train-3.json \
        -valid_dir=${DATA}/coqg-dev-3.json \
        -save_data=${DATA}/processed/coqg.turn3 \
        -data_type=concat \
        -dynamic_dict \
        -src_vocab_size=100000 \
        -tgt_vocab_size=100000 \
        -share_vocab \
        -seq_length=500