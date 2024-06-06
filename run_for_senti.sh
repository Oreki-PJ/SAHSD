CONFIG=config/SBIC_bert_base_for_senti.yml
SEED_LIST=(9 99 999 9999 11 264 8758 5437 4676 2364)
TRAIN_SAMPLE_NUMS_LIST=(16 32 64 128 256 512 1024)


for TRAIN_SAMPLE_NUMS in "${TRAIN_SAMPLE_NUMS_LIST[@]}"
do
  for SEED in "${SEED_LIST[@]}"
  do
    CUDA_VISIBLE_DEVICES=0 python run_for_senti.py \
      --do_test \
      --config $CONFIG \
      --seed $SEED \
      --train_sample_nums $TRAIN_SAMPLE_NUMS
  done
done
