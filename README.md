
<!-- <p align="center" width="100%">
    <a><img src="images/LaMini-LM-solo.png" alt="Title" style="width: 20%; min-width: 300px; display: block; margin: auto;"></a>
</p> -->

## uDistil-Whisper: Label-Free Data Filtering for Knowledge Distillation in Low-Data Regimes ( NAACL'2025 )

<p align="center"><a href="https://macabdul9.github.io/" target="_blank">Abdul Waheed</a>, <a href="https://www.linkedin.com/in/karima-kadaoui-960923b7/" target="_blank">Karima Kadaoui</a>,  <a href="https://scholar.google.com/citations?user=IWcGY98AAAAJ&hl=en"> Bhiksha Raj</a>, <a href="https://mageed.arts.ubc.ca/" target="_blank">Muhammad Abdul-Mageed</a></p>
<p align="center" float="left">
  <img src="assets/cmu_logo.png" height="40" />
  <img src="assets/ubc_logo.png" height="40" />
  <img src="assets/MBZUAI-logo.png" height="40" />
</p>


**Abstract**: Recent work on distilling Whisper's knowledge into small models using pseudo-labels shows promising performance while reducing the size by up to 50\%. This results in small, efficient, and dedicated models. However, a critical step of distillation using pseudo-labels involves filtering high-quality predictions and using only those during training. This step requires ground truth labels to compare with and filter low-quality examples, making the process supervised. Additionally, the distillation process requires a large amount of data thereby limiting its applicability in low-resource settings. To address this, we propose a distillation framework that does not require any labeled data. Through experimentation, we show that our best distilled models outperform the teacher model by 5-7 WER points and are on par with or outperform similar supervised data filtering setups. When scaling the data, our models significantly outperform all zero-shot and supervised models. They are also 25-50\% more compute- and memory-efficient while maintaining performance equal to or better than that of the teacher model.

<!-- TOC -->
**Table of Contents**

1. [Running Evaluation](#running-evaluation)
2. [Models](#models)
3. [Results](#results)
4. [Data](#data)
5. [License Information](#license-information)
6. [Citation](#citation)
7. [Contact](#contact)



<!-- /TOC -->


<!-- I want text in red -->
<!-- > [!CAUTION]
> Note: Do NOT publish OpenBible data as it is behind the LDC paywall and we do not have authority to share the data publicly. https://www.ldc.upenn.edu/data-management/using-data/user-agreements/iarpa-swahili
 -->

### Running Evaluation 

1. Running `openai/whisper-large-v2` on `SADA2022` test set. 
```bash
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
```
Expected output: 
```
{
  "WER": 78.32,
  "CER": 62.15,
  "model_id": "whisper-large-v2",
  "dataset_id": "SADA2022-eval",
  "split": "test",
  "num_samples": 1000
}

```
2. Running our distilled models on `SADA2022` test set.  Please note that the distilled model below is ~50% smaller than the original `whisper-large-v2` model and trained without any labelled data. 

```bash
python evaluate.py \
    --model_name_or_path UBC-NLP/distil-large-v2-init-16-16-100K-225-sm4t \
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
```
Expected output: 
```
{
  "WER": 68.64,
  "CER": 46.21,
  "model_id": "distil-large-v2-init-16-16-100K-225-sm4t",
  "dataset_id": "SADA2022-eval",
  "split": "test",
  "num_samples": 1000
}

```

### Models
#### Main Models
| Model                | Path                                                                                                           |
|----------------------|----------------------------------------------------------------------------------------------------------------|
| UDW-16-16-100K-sonar  | [distil-large-v2-init-16-16-100K-225-sonar_sim](https://huggingface.co/UBC-NLP/distil-large-v2-init-16-16-100K-225-sonar_sim) |
| UDW-16-16-100K-proxy  | [distil-large-v2-init-16-16-100K-225-sm4t](https://huggingface.co/UBC-NLP/distil-large-v2-init-16-16-100K-225-sm4t)       |
| UDW-32-16-100K-sonar  | [distil-large-v2-init-32-16-100K-225-sonar_sim](https://huggingface.co/UBC-NLP/distil-large-v2-init-32-16-100K-225-sonar_sim) |
| UDW-32-16-100K-proxy  | [distil-large-v2-init-32-16-100K-225-sm4t](https://huggingface.co/UBC-NLP/distil-large-v2-init-32-16-100K-225-sm4t)       |
| UDW-16-16-500K-sonar  | [distil-large-v2-init-16-16-500K-225-sonar_sim](https://huggingface.co/UBC-NLP/distil-large-v2-init-16-16-500K-225-sonar_sim) |
| UDW-16-16-500K-proxy  | [distil-large-v2-init-16-16-500K-225-sm4t](https://huggingface.co/UBC-NLP/distil-large-v2-init-16-16-500K-225-sm4t)       |
| UDW-32-16-500K-sonar  | [distil-large-v2-init-32-16-500K-225-sonar_sim](https://huggingface.co/UBC-NLP/distil-large-v2-init-32-16-500K-225-sonar_sim) |
| UDW-32-16-500K-proxy  | [distil-large-v2-init-32-16-500K-225-sm4t](https://huggingface.co/UBC-NLP/distil-large-v2-init-32-16-500K-225-sm4t)       |
					
					

<span style="color: orange;">**Note**.</span> Models trained on 100K examples are labeled as 100K here, while those trained on 500K examples are labeled as 500K. In the paper, however, these models are referred to as UDW-{nenc}-{ndec} for the 100K configuration and UDW-{nenc}-{ndec}++ for the 500K configuration, followed by the specific method (e.g., proxy, sonar), where nenc and ndec denote the number of encoder and decoder blocks, respectively.


### Results

 
<!-- Add image -->
<image src="assets/main_results.png" alt="Results" style="width: 100%; min-width: 300px; display: block; margin: auto;">



<!-- Add other_results.png -->
<image src="assets/other_results.png" alt="Results" style="width: 100%; min-width: 300px; display: block; margin: auto;">

<!-- #### Swahili Results

<!-- Swahili Results saved at assets/results_swahili.png -->
<!-- <image src="assets/results_swahili.png" alt="Results" style="width: 100%; min-width: 300px; display: block; margin: auto;"> -->
 <!-- -->

### Data
### Training Dataset
### Training Dataset
| Dataset                                     | Path                                                                                                                  |
|---------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| Dataset-100K    | [masc_cv15_asc_fleurs_mgb5_mgb2_qasr_100K](https://huggingface.co/datasets/UBC-NLP/masc_cv15_asc_fleurs_mgb5_mgb2_qasr_100K)    |
| Dataset-500K    | [masc_cv15_asc_fleurs_mgb5_mgb2_qasr_500K](https://huggingface.co/datasets/UBC-NLP/masc_cv15_asc_fleurs_mgb5_mgb2_qasr_500K)    |



### Evaluation Dataset
| Dataset | Description |
|---------|-------------|
| [Fleurs](https://huggingface.co/datasets/google/fleurs) | Multilingual speech dataset aimed at evaluating speech-to-text systems. We use the `test` and `validation` splits of the `ar_eg` part. |
| [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_15_0) | A crowdsourced initiative to create a free and publicly available dataset of diverse voices. We use three versions: [CV-6.1](https://huggingface.co/datasets/mozilla-foundation/common_voice_6_1), [CV-9.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_9_0), [CV-11.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0), and [CV-15.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_15_0). |
| [MGB2](https://huggingface.co/datasets/UBC-NLP/MGB2-Eval) | Dataset focused on broadcast news analysis as part of the MGB Challenge. Contains roughly 70% Modern Standard Arabic (MSA) and the remainder in dialects (Egyptian (EGY), Gulf (GLF), Levantine (LEV), and North African (NOR)). Uses the `test` and `validation` splits. |
| [MGB3](https://huggingface.co/datasets/UBC-NLP/MGB3) | Includes 80 programs from Egyptian YouTube channels across various genres, with 4.8 hours of transcribed data from the first 12 minutes of each program. Evaluated on `test` and `validation` splits. |
| [MGB5](https://huggingface.co/datasets/UBC-NLP/MGB5) | Comprises 10.2 hours of Moroccan Arabic speech data from 93 YouTube videos across genres such as comedy, cooking, and sports. Evaluated on `test` and `validation` splits. |
| [SADA2022](https://ieeexplore.ieee.org/document/10446243) | Evaluated on `test` and `validation` splits. |
| [Casablanca](https://aclanthology.org/2024.emnlp-main.1211/) | Evaluated on `test` and `validation` splits. |



### License Information

1.  **Code** - The source code for this project is licensed under [MIT](LICENSE).

2. **Data** - We do not own the data used in this project. The data remain under the original license provided by the data source. Please refer to the original data license for its terms.

3. **Trained Models** - The trained models are distributed under the same license as the original models, subject to any restrictions imposed by the data license. For a complete list of these restrictions, please consult the corresponding section in the original data license.


### Citation 
If you find our work useful, please consider citing our paper and the related works listed below.
 
```
@misc{waheed2024udistilwhisperlabelfreedatafiltering,
      title={uDistil-Whisper: Label-Free Data Filtering for Knowledge Distillation in Low-Data Regimes}, 
      author={Abdul Waheed and Karima Kadaoui and Bhiksha Raj and Muhammad Abdul-Mageed},
      year={2024},
      eprint={2407.01257},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.01257}, 
      publisher = {NAACL2025},
}
```

Related Work:
```
@inproceedings{waheed-etal-2024-distill,
  title = "To Distill or Not to Distill? On the Robustness of Robust Knowledge Distillation",
  author = "Waheed, Abdul and Kadaoui, Karima and Abdul-Mageed, Muhammad",
  booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics",
  year = "2024",
  month = aug,
  pages = "12603--12621",
  url = "https://aclanthology.org/2024.acl-long.680/",
  doi = "10.18653/v1/2024.acl-long.680"
}

```
We ran our experiments based on [Gandhi et. al.](https://arxiv.org/abs/2406.04512) with some modification. If you find our work helpful, please also consider citing their and original Whisper paper.

```
@misc{gandhi2023distilwhisper,
      title={Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling}, 
      author={Sanchit Gandhi and Patrick von Platen and Alexander M. Rush},
      year={2023},
      eprint={2311.00430},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{radford2022robust,
      title={Robust Speech Recognition via Large-Scale Weak Supervision}, 
      author={Alec Radford and Jong Wook Kim and Tao Xu and Greg Brockman and Christine McLeavey and Ilya Sutskever},
      year={2022},
      eprint={2212.04356},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

### Contact
For any queries, please contact Abdul Waheed (abdulwaheed1513@gmail.com) or Karima Kadaoui (karima.kadaoui@mbzuai.ac.ae). 
