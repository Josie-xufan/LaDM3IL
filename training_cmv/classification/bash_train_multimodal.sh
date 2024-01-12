#!/usr/bin/env bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER
export PYTHONPATH=$SHELL_FOLDER
echo "current path " $SHELL_FOLDER

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

# training
echo "CMV Multimodal Training..."
python train_multimodal.py --batch_size 128 \
--updprot_threshold 0.8 \
--prot_start 15 \
--proto_ema 0.810 \
--vocab_path ../../pretrained/vocab_3mer.pkl \
--gene_token ../../dataset/cmv/train_val_test/gene.csv \
--train_dataset ../../dataset/cmv/train_val_test/train.tsv \
--valid_dataset ../../dataset/cmv/train_val_test/val.tsv \
--test_dataset ../../dataset/cmv/train_val_test/test.tsv \
--output_path ../../result/cmv

# inference
echo "CMV Multimodal Inference..."
python train_multimodal.py --batch_size 128 \
--updprot_threshold 0.8 \
--prot_start 15 \
--proto_ema 0.810 \
--vocab_path ../../pretrained/vocab_3mer.pkl \
--gene_token ../../dataset/cmv/train_val_test/gene.csv \
--train_dataset ../../dataset/cmv/train_val_test/train.tsv \
--valid_dataset ../../dataset/cmv/train_val_test/val.tsv \
--test_dataset ../../dataset/cmv/train_val_test/test.tsv \
--output_path ../../result/cmv \
--reuse_ckpt_dir ../../pretrained/cmv_classification/max_auc_model.pth
