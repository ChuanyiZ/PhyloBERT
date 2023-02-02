export KMER=6
export SOURCE="/home/ac.chia/"
export TRAIN_FILE=$SOURCE"/PhyloBERT/data/6mer_cut6.txt.gz"
export TEST_FILE=$TRAIN_FILE
export OUTPUT_PATH="/home/ac.chia/PhyloBERT/models/pretrain_half_mlm_only"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 run_pretrain.py \
    --output_dir $OUTPUT_PATH \
    --model_type=dna \
    --tokenizer_name=$SOURCE"PhyloBERT/example/data/6-new-12w-0" \
    --config_name=$SOURCE"PhyloBERT/example/data/6-new-12w-0/config_half.json" \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --gradient_accumulation_steps 25 \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 64 \
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
