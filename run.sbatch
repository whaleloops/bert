#!/bin/sh
#
# run.sbatch
#
#SBATCH --job-name=yangbert1
#SBATCH --output=yangbert1.txt           # output file
##SBATCH -n 4                          # Number of cores
#SBATCH -N 1                           # Ensure that all cores are on one machine
#SBATCH -p titanx-short                # Partition to submit to (serial_requeue)
#SBATCH --mem=20000                      # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -e yangbert1.err             # File to which STDERR will be written
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=zhichaoyang@cs.umass.edu # Email to which notifications will be sent
#SBATCH --gres=gpu:1

cd /home/zhanxu/yangzhic/bert/bert
sleep 1
export BERT_BASE_DIR=../uncased_L-12_H-768_A-12
source activate bertenv
sleep 1
python run_pretraining.py \
  --input_file=../tmp/clinic_train.tfrecord \
  --output_dir=../tmp/pretraining_output1 \
  --do_train=True \
  --do_eval=True \
  --do_predict=False\
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=30 \
  --eval_batch_size=30 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=2000\
  --num_warmup_steps=200 \
  --save_checkpoints_steps=500 \
  --learning_rate=2e-6
sleep 1
exit