export KMER=6
export SOURCE="/home/chuanyi/project/phylobert/"
export TRAIN_FILE=$SOURCE"/PhyloBERT/example/data/6_3k.txt"
export TEST_FILE=$TRAIN_FILE
export OUTPUT_PATH="/home/jupyter/models/pretrain"

python run_pretrain.py \
    --output_dir $OUTPUT_PATH \
    --model_type=dna \
    --tokenizer_name=$SOURCE"/PhyloBERT/example/data/6-new-12w-0" \
    --config_name=$SOURCE"/PhyloBERT/example/data/6-new-12w-0/config.json" \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --gradient_accumulation_steps 25 \
    --per_gpu_train_batch_size 10 \
    --per_gpu_eval_batch_size 6 \
    --save_steps 500 \
    --save_total_limit 20 \
    --max_steps 200000 \
    --evaluate_during_training \
    --logging_steps 500 \
    --line_by_line \
    --learning_rate 4e-4 \
    --block_size 512 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.98 \
    --mlm_probability 0.025 \
    --warmup_steps 10000 \
    --overwrite_output_dir \
    --n_process 4