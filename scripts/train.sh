#!/usr/bin/env bash

PROJ=/research/king3/yfgao/pd/conversational-QG
PYT=/research/king2/yfgao/envs/pyt0.4.1py3.6cuda9/bin/python

export CUDA_VISIBLE_DEVICES=$1
HISTORY_TURN=3
DATE=$2
MODEL_NAME=$3
LCV=1
LCA=1
LF=1
LFH=1

${PYT} -u code/train_single.py \
        -word_vec_size=300 \
        -share_embeddings \
        -feat_vec_size=20 \
        -data_type=concat \
        -encoder_type=brnn \
        -passage_enc_layers=2 \
        -qa_word_enc_layers=2 \
        -qa_sent_enc_layers=1 \
        -copy_attn \
        -reuse_copy_attn \
        -coref_vocab \
        -lambda_coref_vocab=${LCV} \
        -coref_attn \
        -lambda_coref_attn=${LCA} \
        -flow \
        -lambda_flow=${LF} \
        -flow_history \
        -lambda_flow_history=${LFH} \
        -data=${PROJ}/data/processed/coqg.turn${HISTORY_TURN} \
        -save_model=${PROJ}/data/model/${DATE}_turn${HISTORY_TURN}_${MODEL_NAME}_lcv_${LCV}_lca_${LCA}_lf${LF}_lfh_${LFH} \
        -save_checkpoint_steps=2000 \
        -keep_checkpoint=4 \
        -gpuid=0 \
        -pre_word_vecs_enc=${PROJ}/data/processed/vocab.glove.840B.300d.enc.pt \
        -pre_word_vecs_dec=${PROJ}/data/processed/vocab.glove.840B.300d.enc.pt \
        -batch_size=32 \
        -valid_steps=2000 \
        -valid_batch_size=16 \
        -train_steps=20000 \
        -optim=adagrad \
        -adagrad_accumulator_init=0.1 \
        -learning_rate=0.1 \
        -learning_rate_decay=0.5 \
        -start_decay_steps=10001 \
        -decay_steps=2000 \
        -report_every=200