## Conversational Question Generation

Implementation for our ACL 2019 paper: Interconnected Question Generation with Coreference Alignment and Conversation Flow Modeling

### Requirements
tqdm;
pytorch=0.4.1;
torchtext;
numpy;
python=3.6;
cuda=9.0


### Download Processed Data
Please clone this repo, download our processed data [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155102332_link_cuhk_edu_hk/EdOnhcRyjP5LiqDDM4_IK2oBDe91LL0plw16SBQkEbV39Q?e=zCdcn4) and put it into `data` directory:

```bash
git clone https://github.com/Evan-Gao/conversaional-QG.git
cd conversational-QG
mkdir data
```

coqg-train/dev/test-3.json: our train/dev/test data split

coqg-coref-test-3.json: coreference test set

### Preprocess
run `scripts/preprocess.sh` for preprocessing.

GloVe vectors are required, please download [glove.840B.300d](http://nlp.stanford.edu/data/glove.840B.300d.zip) first.
run `scripts/emb.sh` for getting corresponding word embedding.

### Train, Generate \& Evaluate
run `scripts/train.sh` for training, `scripts/generate.sh` for generation and evaluation

### Pretrained Model
We have released our pretrained model [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155102332_link_cuhk_edu_hk/EQNpN00ivPNOm0LDMASZO7IBDrzzgeMDGZD9Z319GRQ76Q?e=82d2mr).

### Reference
If you use code, please cite our paper as follows:

```tex
@inproceedings{Gao2019InterconnectedQG,
	title="Interconnected Question Generation with Coreference Alignment and Conversation Flow Modeling",
	author="Yifan Gao and Piji Li and Irwin King and Michael R. Lyu",
	booktitle="ACL",
	year="2019"
}
```