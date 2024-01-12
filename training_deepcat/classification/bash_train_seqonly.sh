#!/usr/bin/env bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER
export PYTHONPATH=$SHELL_FOLDER
echo "current path " $SHELL_FOLDER

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

# training
echo "Deepcat Seqonly Training..."
python train_seqonly.py \
--batch_size 16 \
--updprot_threshold 0.9 \
--prot_start 15 \
--proto_ema 0.963 \
--train_dataset ../../dataset/deepcat/train_val_test/train.tsv \
--valid_dataset ../../dataset/deepcat/train_val_test/val.tsv \
--test_dataset ../../dataset/deepcat/train_val_test/test.tsv \
--output_path ../../result/deepcat

# inference
echo "Deepcat Seqonly Inference..."
python train_seqonly.py \
--batch_size 16 \
--updprot_threshold 0.9 \
--prot_start 15 \
--proto_ema 0.963 \
--train_dataset ../../dataset/deepcat/train_val_test/train.tsv \
--valid_dataset ../../dataset/deepcat/train_val_test/val.tsv \
--test_dataset ../../dataset/deepcat/train_val_test/test.tsv \
--output_path ../../result/deepcat \
--reuse_ckpt_dir ../../pretrained/deepcat_classification/max_auc_model.pth
