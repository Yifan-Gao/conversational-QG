#!/usr/bin/env bash

PROJ=/research/king3/yfgao/pd/conversational-QG
PYT=/research/king2/yfgao/envs/pyt0.4.1py3.6cuda9/bin/python
PYT2=/bin/python

export CUDA_VISIBLE_DEVICES=$1
HISTORY_TURN=3
DATE=$2
MODEL_NAME=$3
LCV=1
LCA=1
LF=1
LFH=1
N_STEP=$4

${PYT} -u code/translate.py \
        -model=${PROJ}/data/model/${DATE}_turn${HISTORY_TURN}_${MODEL_NAME}_lcv_${LCV}_lca_${LCA}_lf${LF}_lfh_${LFH}_step_${N_STEP}.pt \
        -data=${PROJ}/data/coqg-test-${HISTORY_TURN}.json \
        -output=${PROJ}/data/pred/${DATE}_turn${HISTORY_TURN}_${MODEL_NAME}_lcv${LCV}_lca_${LCA}_lf${LF}_lfh_${LFH}_step_${N_STEP}_test.txt \
        -dynamic_dict \
        -share_vocab \
        -block_ngram_repeat=1 \
        -replace_unk \
        -batch_size=2 \
        -gpu=0

${PYT2} -u ${PROJ}/code/tools/eval/eval.py \
        -out=${PROJ}/data/pred/${DATE}_turn${HISTORY_TURN}_${MODEL_NAME}_lcv${LCV}_lca_${LCA}_lf${LF}_lfh_${LFH}_step_${N_STEP}_test.txt \
        -tgt=${PROJ}/data/coqg-test-tgt-3.txt
