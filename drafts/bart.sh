DATA_ROOT_DIR=../../WebSRC/data
CUDA_VISIBLE_DEVICES=0

python -u -W ignore bart.py                             \
  --root_dir ${DATA_ROOT_DIR}                           \
  --train_file ${DATA_ROOT_DIR}/websrc1.0_train_.json   \
  --eval_file ${DATA_ROOT_DIR}/websrc1.0_dev_.json      \
  --backbone baseline --method baseline                 \
  --pretrained_model_name_or_path ./tmp_model/bart-base-2/\
  --output_dir tmp_result/                              \
  --run_train                                           \
  --eval_when_train                                     \
  --num_training_epochs 10                              \
  --per_gpu_train_batch_size 8                          \
  --overwrite_output_dir