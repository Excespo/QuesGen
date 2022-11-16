BACKBONE=baseline
METHOD=baseline
SEED=42
NUM_TRAINING_EPOCHS=10

DATA_ROOT_DIR=../../WebSRC/data
CUDA_VISIBLE_DEVICES=0
OUTPUT_DIR=tmp_result_${BACKBONE}_${METHOD}_${SEED}_${NUM_TRAINING_EPOCHS}

python -u -W ignore bart.py                             \
  --root_dir ${DATA_ROOT_DIR}                           \
  --train_file ${DATA_ROOT_DIR}/websrc1.0_train_.json   \
  --eval_file ${DATA_ROOT_DIR}/websrc1.0_dev_.json      \
  --backbone ${BACKBONE} --method ${METHOD}             \
  --pretrained_model_name_or_path ./tmp_model/bart-base-2/\
  --output_dir tmp_result/                              \
  --run_train                                           \
  --eval_when_train                                     \
  --num_training_epochs ${NUM_TRAINING_EPOCHS}          \
  --per_gpu_train_batch_size 8                          \
  --overwrite_output_dir