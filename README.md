
<!-- <p align="center" width="100%">
    <a><img src="images/LaMini-LM-solo.png" alt="Title" style="width: 20%; min-width: 300px; display: block; margin: auto;"></a>
</p> -->

# uDistil-Whisper: Label-Free Data Filtering for Knowledge Distillation in Low-Data Regimes

<a href="https://macabdul9.github.io/" target="_blank">Abdul Waheed</a>, <a href="Karima Kadaoui
" target="_blank">Karima Kadoui</a>,  <a href="https://scholar.google.com/citations?user=IWcGY98AAAAJ&hl=en"> Bhiksha Raj</a> <a href="https://mageed.arts.ubc.ca/" target="_blank">Muhammad Abdul-Mageed</a></p>
<p align="center" float="left">
  <img src="assets/cmu_logo.png" height="40" />
  <img src="assets/ubc_logo.png" height="40" />
  <img src="assets/MBZUAI-logo.png" height="40" />
</p>




### Models
#### Main Models

| Model                                         | Path                                                                                                           |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| distil-large-v2-init-8-8-100K-225             | [distil-large-v2-init-8-8-100K-225](https://huggingface.co/UBC-NLP/distil-large-v2-init-8-8-100K-225)           |								
| distil-large-v2-init-16-16-100K-225           | [distil-large-v2-init-16-16-100K-225](https://huggingface.co/UBC-NLP/distil-large-v2-init-16-16-100K-225)       |								
| distil-large-v2-init-16-16-500K-225           | [distil-large-v2-init-16-16-500K-225](https://huggingface.co/UBC-NLP/distil-large-v2-init-16-16-500K-225)       |								
| distil-large-v2-init-16-32-100K-225           | [distil-large-v2-init-16-32-100K-225](https://huggingface.co/UBC-NLP/distil-large-v2-init-16-32-100K-225)       |								
| distil-large-v2-init-32-16-100K-225           | [distil-large-v2-init-32-16-100K-225](https://huggingface.co/UBC-NLP/distil-large-v2-init-32-16-100K-225)       |								
| distil-large-v2-init-32-16-500K-225           | [distil-large-v2-init-32-16-500K-225](https://huggingface.co/UBC-NLP/distil-large-v2-init-32-16-500K-225)       |								



### Training Dataset
| Dataset                                     | Path                                                                                                                  |
|---------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| Dataset-100K    | [masc_cv15_asc_fleurs_mgb5_mgb2_qasr_100K](https://huggingface.co/datasets/UBC-NLP/masc_cv15_asc_fleurs_mgb5_mgb2_qasr_100K)    |
| Dataset-500K    | [masc_cv15_asc_fleurs_mgb5_mgb2_qasr_500K](https://huggingface.co/datasets/UBC-NLP/masc_cv15_asc_fleurs_mgb5_mgb2_qasr_500K)    |
| Dataset-1M      | [masc_cv15_asc_fleurs_mgb5_mgb2_qasr_1M](https://huggingface.co/datasets/UBC-NLP/masc_cv15_asc_fleurs_mgb5_mgb2_qasr_1M)          |



### Evaluation Dataset
| Dataset                                           | Description                                                                                   |
|---------------------------------------------------|-----------------------------------------------------------------------------------------------|
| [Fleurs](https://huggingface.co/datasets/google/fleurs)           | Multilingual speech dataset aimed at evaluating speech-to-text systems. We use `test` and `validation` split of `ar_eg` part.                     |
| [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_15_0) | A crowdsourced initiative to create a free and publicly available dataset of diverse voices.We use three versions of Common Voice, [CV-6.1](https://huggingface.co/datasets/mozilla-foundation/common_voice_6_1), [CV-9.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_9_0), [CV-11.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0), and [CV-15.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_15_0) |
| [MGB2](https://huggingface.co/datasets/UBC-NLP/MGB2-Eval)               | Dataset focused on broadcast news analysis, part of the MGB Challenge, which contains rougly 70% MSA and rest are dialects mainly - Egyptian (EGY), Gulf (GLF), Lev- antine (LEV), and North African (NOR) . We use `test` and `validation` split.                        |
| [MGB3](https://huggingface.co/datasets/UBC-NLP/MGB3)               | This dataset includes 80 programs from Egyptian YouTube channels across various genres, with 4.8 hours of transcribed data from the first 12 minutes of each program. We evaluate our models on `test` and `validation` split.                      |
| [MGB5](https://huggingface.co/datasets/UBC-NLP/MGB5)               | This dataset includes 10.2 hours of Moroccan Arabic speech data from 93 YouTube videos across seven genres like comedy, cooking, and sports. We evaluate our models on `test` and `validation` split.              |


### Results

### Training 
We train our models using the codebase developed by [Gandhi et al.](https://github.com/huggingface/distil-whisper) and express our gratitude for their open-sourcing it. After processing our data into the required format, we train our models in three steps. Please follow the steps below for training/distillation.

0. **Clone**  `distil-whisper` [fork repo](https://github.com/macabdul9/distil-whisper) and change the directory.

```
git clone git@github.com:macabdul9/distil-whisper.git && cd distil-whisper
```

1. **pip install the required packages from the** `setup.py` **file:**
```
cd training
pip install -e .
cd ..
```

2. **Student Initialization**: ```bash scripts/student_initialization.sh```
OR:
```
python training/create_student_model.py \
  --teacher_checkpoint "openai/whisper-large-v2" \
  --encoder_layers 32 \
  --decoder_layers 16 \
  --save_dir "./models/init-32-16"
```

3. **Psudo Labelling**: ```bash scripts/pseudo_labelling.sh``` OR:
```
accelerate launch training/run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v2" \
  --dataset_name "masc_cv15_asc_fleurs_mgb5_mgb2_qasr_100K" \
  --load_from_disk False \
  --max_samples_per_split 1000 \
  --text_column_name "text" \
  --audio_column_name "audio" \
  --id_column_name "id" \
  --output_dir "./data/masc_cv15_asc_fleurs_mgb5_mgb2_qasr_pseudo_labelled-v2-100K-225" \
  --wandb_project "distil-whisper-labelling" \
  --per_device_eval_batch_size 80 \
  --dtype "bfloat16" \
  --dataloader_num_workers 8 \
  --preprocessing_num_workers 8 \
  --logging_steps 500 \
  --max_label_length 225 \
  --language "ar" \
  --task "transcribe" \
  --return_timestamps \
  --streaming False \
  --generation_num_beams 1 \
  --decode_token_ids False \
```

4. **Distillation/Training**: ```bash scripts/distillation.sh``` OR:
```
torchrun --standalone --nnodes=1 --nproc-per-node=8 training/run_distillation.py \
    --model_name_or_path "./models/init-32-16" \
    --teacher_model_name_or_path "openai/whisper-large-v2" \
    --output_dir "models/distil-large-v2-init-32-16-100K-225" \
    --train_dataset_name "./data/masc_cv15_asc_fleurs_mgb5_mgb2_qasr_pseudo_labelled-v2-100K-225" \
    --train_split_name "train" \
    --eval_split_name "test" \
    --text_column_name "text" \
    --max_train_samples 1000 \
    --max_eval_samples 1000 \
    --save_steps 1000 \
    --warmup_steps 50 \
    --learning_rate 0.0001 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 25 \
    --save_total_limit 1 \
    --num_train_epochs 10 \
    --wer_threshold 80 \
    --max_label_length 225 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 4 \
    --preprocessing_batch_size 1000 \
    --ddp_timeout 28800 \
    --dtype "bfloat16" \
    --do_train \
    --gradient_checkpointing \
    --predict_with_generate \
    --streaming False \
    --overwrite_output_dir \

```

**Note**: Pseudo labelling and training is limited to 1000 examples for quick dry run. Please remove/unset the relevant agrs to run for whole dataset.

### Citation 
Please use the following bibtex to cite our work. 
```
@article{waheed2024udistil,
  title={uDistil-Whisper: Label-Free Data Filtering for Knowledge Distillation via Large-Scale Pseudo Labelling},
  author={Waheed, Abdul and Kadaoui, Karima and Abdul-Mageed, Muhammad},
  journal={arXiv preprint arXiv:2407.01257},
  year={2024}
}
```


If you find our work useful please consider citing the following related work as well. 
```
@misc{waheed2024distill,
      title={To Distill or Not to Distill? On the Robustness of Robust Knowledge Distillation}, 
      author={Abdul Waheed and Karima Kadaoui and Muhammad Abdul-Mageed},
      year={2024},
      eprint={2406.04512},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
We ran our experiments based on [Gandhi et. al.](https://arxiv.org/abs/2406.04512) with some modification. We recommend to cite this work if you find our work useful. 

```
@misc{gandhi2023distilwhisper,
      title={Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling}, 
      author={Sanchit Gandhi and Patrick von Platen and Alexander M. Rush},
      year={2023},
      eprint={2311.00430},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```

We also request you to cite original Whisper paper.
```
@misc{radford2022robust,
      title={Robust Speech Recognition via Large-Scale Weak Supervision}, 
      author={Alec Radford and Jong Wook Kim and Tao Xu and Greg Brockman and Christine McLeavey and Ilya Sutskever},
      year={2022},
      eprint={2212.04356},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

### Acknowledgement
