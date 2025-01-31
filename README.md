
<!-- <p align="center" width="100%">
    <a><img src="images/LaMini-LM-solo.png" alt="Title" style="width: 20%; min-width: 300px; display: block; margin: auto;"></a>
</p> -->

# uDistil-Whisper: Label-Free Data Filtering for Knowledge Distillation in Low-Data Regimes

<p align="center"><a href="https://macabdul9.github.io/" target="_blank">Abdul Waheed</a>, <a href="Karima Kadaoui
" target="_blank">Karima Kadoui</a>,  <a href="https://scholar.google.com/citations?user=IWcGY98AAAAJ&hl=en"> Bhiksha Raj</a> <a href="https://mageed.arts.ubc.ca/" target="_blank">Muhammad Abdul-Mageed</a></p>
<p align="center" float="left">
  <img src="assets/cmu_logo.png" height="40" />
  <img src="assets/ubc_logo.png" height="40" />
  <img src="assets/MBZUAI-logo.png" height="40" />
</p>

<!-- TOC -->

- [Models](#models)
- [Evaluation](#evaluation)
- [Results](#results)
- [Data](#data)
- [Citation](#citation)

<!-- /TOC -->


<!-- I want text in red -->
<font color='red'>Note: Do not public OpenBible data as it is behind LDC paywall and we don't have authority to share the data publically. https://www.ldc.upenn.edu/data-management/using-data/user-agreements/iarpa-swahili.</font>


## Models
#### Main Models

| Model                                         | Path                                                                                                           |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| distil-large-v2-init-8-8-100K-225             | [distil-large-v2-init-8-8-100K-225](https://huggingface.co/UBC-NLP/distil-large-v2-init-8-8-100K-225)           |								
| distil-large-v2-init-16-16-100K-225           | [distil-large-v2-init-16-16-100K-225](https://huggingface.co/UBC-NLP/distil-large-v2-init-16-16-100K-225)       |								
| distil-large-v2-init-16-16-500K-225           | [distil-large-v2-init-16-16-500K-225](https://huggingface.co/UBC-NLP/distil-large-v2-init-16-16-500K-225)       |								
| distil-large-v2-init-16-32-100K-225           | [distil-large-v2-init-16-32-100K-225](https://huggingface.co/UBC-NLP/distil-large-v2-init-16-32-100K-225)       |								
| distil-large-v2-init-32-16-100K-225           | [distil-large-v2-init-32-16-100K-225](https://huggingface.co/UBC-NLP/distil-large-v2-init-32-16-100K-225)       |								
| distil-large-v2-init-32-16-500K-225           | [distil-large-v2-init-32-16-500K-225](https://huggingface.co/UBC-NLP/distil-large-v2-init-32-16-500K-225)       |								

### Evaluation
```bash
python evaluate.py --model_name_or_path UBC-NLP/distil-large-v2-init-8-8-100K-225 --dataset_name google/fleurs --split test
```



### Results
 
<!-- Add image -->
<image src="assets/main_results.png" alt="Results" style="width: 100%; min-width: 300px; display: block; margin: auto;">

<!-- Add other_results.png -->
<image src="assets/other_results.png" alt="Results" style="width: 100%; min-width: 300px; display: block; margin: auto;">

<!-- #### Swahili Results

<!-- Swahili Results saved at assets/results_swahili.png -->
<image src="assets/results_swahili.png" alt="Results" style="width: 100%; min-width: 300px; display: block; margin: auto;"> -->


## Data
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

#### Swahili Dataset
### Training Data

| Dataset                                   | Description                                                                                   |
|-------------------------------------------|-----------------------------------------------------------------------------------------------|
| UBC-NLP/BabelSwahili-Scripted             | Scripted Swahili dataset                                                                      |
| UBC-NLP/BabelSwahili-Conversation         | Conversational Swahili dataset                                                               |
| UBC-NLP/AMMI-LigAikuma-Swahili            | Swahili dataset from the LigAikuma project                                                   |
| UBC-NLP/DVoice_Swahili                    | Swahili voice dataset                                                                         |

### Evaluation Data

| Dataset                                   | Description                                                                                   |
|-------------------------------------------|-----------------------------------------------------------------------------------------------|
| UBC-NLP/OpenBible_Swahili-Eval            | Swahili evaluation dataset from the OpenBible project                                         |
| UBC-NLP/CommonVoice17_Swahili-Eval        | Evaluation dataset for Swahili from CommonVoice17                                             |
| UBC-NLP/ALFAA_Swahili-Eval                | Evaluation dataset for Swahili from the ALFAA project                                         |
| UBC-NLP/BabelSwahili-Conversation-Eval    | Conversational Swahili evaluation dataset                                                    |
| UBC-NLP/Fleurs_Swahili-eval               | Evaluation dataset for Swahili from the Fleurs project                                        |



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

### Licence

### Contact
For any queries, please contact Abdul Waheed (abdulwaheed1513@gmai.com) or Karima Kadaoui ( email@karima.com). 

