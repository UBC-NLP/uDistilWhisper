python evaluate.py \
    --model_name_or_path openai/whisper-large-v2 \
    --dataset_name_or_path UBC-NLP/SADA2022-eval \
    --config default \
    --split test \
    --audio_column audio \
    --text_column raw_transcript \
    --device cuda:0 \
    --num_samples 1000 \
    --num_proc 4 \
    --do_normalize \
    --output_dir outputs

python evaluate.py \
    --model_name_or_path macabdul9/distil-large-v2-init-16-16-100K-225-sm4t \
    --dataset_name_or_path UBC-NLP/SADA2022-eval \
    --config default \
    --split test \
    --audio_column audio \
    --text_column raw_transcript \
    --device cuda:0 \
    --num_samples 1000 \
    --num_proc 4 \
    --do_normalize \
    --output_dir outputs
