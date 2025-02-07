python evaluate.py \
    --model_name_or_path openai/whisper-large-v2 \
    --dataset_name_or_path google/fleurs \
    --config ar_eg \
    --split test \
    --audio_column audio \
    --text_column transcription \
    --do_normalize \
    --device mps \