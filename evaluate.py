# Author : Abdul Waheed

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset
from tqdm import tqdm
import functools
import jiwer
import json 
import librosa
import re
import pandas as pd
import os

def preprocess(text):
    # Based on https://arxiv.org/pdf/2105.14779.pdf

    # 1- remove the sepecial cahrachters and diacritics
    #   -- We did NOT convert the Latin characters to lower case as we are not working on Code-Switch dataset, so we removed all the Latin letters.
    text = re.sub(r"[^0-9\u0621-\u064A\u0660-\u0669%@\s]", "", text)
    # text = re.sub(r"[^0-9\u0600-\u06FF\u0660-\u0669%@\s]", "", text) #065E # include diacritics

    # 2- transliterating all Arabic digits to Arabic numerals (We used Whisper to verify the numerical output)
    text = re.sub(r"[٠-٩]",lambda x: str(int(x.group())), text)

    # 3- Normalize alefs
    text = re.sub("[إأٱآا]", "ا", text)

    # For Haa nad Taa, we didn't see a problem in CV9.0 data. (We need to discuss this further)

    # - Remove extra spaces
    text = " ".join(text.split())
    
    return text

def resample_audio(example, audio_column="audio", target_sample_rate=16000):
    """
    Resample the audio to the target sample rate.

    Args:
        example (dict): The example containing the audio.
        target_sample_rate (int): The target sample rate.
        return example (dict): The example containing the resampled audio.
    """
    # Resample the audio with librosa
    example[audio_column]["array"] = librosa.resample(y=example[audio_column]["array"], orig_sr=example[audio_column]["sampling_rate"], target_sr=target_sample_rate)
    
    example[audio_column]["sampling_rate"] = target_sample_rate
    return example
    

def compute_metrics(ground_truths, hypothesis):
    """
    Compute WER, CER, total errors, and reference length.

    Args:
        ground_truth (List): The reference text.
        hypothesis (List): The generated text.

    Returns:
        dict: A dictionary with WER, CER, total errors, and reference length.
    """
    # Normalize texts

    # Compute the WER and CER
    WER = jiwer.wer(ground_truths, hypothesis)
    CER = jiwer.cer(ground_truths, hypothesis)
    
    # scaling the WER and CER so that the scale is same as o and ncErrors

    return {
        "WER": round(WER*100, 2),
        "CER": round(CER*100, 2),
    }

def load_model(model_name_or_path, device="auto"):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path, device_map=device)
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
        
    return {"model": model, "processor": processor}

def transcribe_audio(model_dict, audio, sampling_rate=16_000):
    
    processor = model_dict["processor"]
    model = model_dict["model"]
    
    input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(model.device)
    
    with torch.no_grad():
        pred_ids = model.generate(input_features, forced_decoder_ids=model.config.forced_decoder_ids)

    output = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
    return output


def main(args):
    
    # Set seed for everything - reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Load the dataset and the model
    dataset = load_dataset(args.dataset_name_or_path, args.config, split=args.split)
    
    # sample the dataset
    if args.num_samples is not None:
        dataset = dataset.shuffle(seed=args.seed).select(range(min(args.num_samples, len(dataset))))
    
    # resample function
    resample_fn = functools.partial(resample_audio, audio_column=args.audio_column, target_sample_rate=args.sample_rate)
    
    # resample the audio
    dataset = dataset.map(resample_fn, num_proc=args.num_proc)
    
    # Load the model
    model_dict = load_model(args.model_name_or_path, device=args.device)
    
    # Iterate over the dataset and transcribe the audio
    hypothesis = []
    ground_truths = []
    for example in tqdm(dataset):
        
        # Input speech
        audio = example[args.audio_column]['array']
        output = transcribe_audio(model_dict=model_dict, audio=audio)
        
        # Append the hypothesis and the ground truth
        hypothesis.append(output)
        ground_truths.append(example[args.text_column])
        
    # Normalize the texts
    df = pd.DataFrame({"ground_truths": ground_truths, "hypothesis": hypothesis})
    
    if args.do_normalize:
        df["ground_truths"] = df["ground_truths"].apply(preprocess)
        df["hypothesis"] = df["hypothesis"].apply(preprocess)
    
    
    # Compute the metrics
    metrics = compute_metrics(ground_truths=df["ground_truths"].values.tolist(), hypothesis=df["hypothesis"].values.tolist())
    
    model_id = args.model_name_or_path.split("/")[-1]
    dataset_id = args.dataset_name_or_path.split("/")[-1]
    metrics["model_id"] = model_id
    metrics["dataset_id"] = dataset_id
    metrics["split"] = args.split
    metrics["num_samples"] = len(dataset)
    
    
    # Print the metrics
    print(metrics)
    
    # Save the metrics
    
    if args.output_dir is not None:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{model_id}_{dataset_id}_{args.split}_metrics.json"
        with open(output_path, "w") as f:
            json.dump(metrics, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate the uDistilWhisper models.')
    
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-large-v2", help="The model checkpoint for weights initialization. It should be either a local path or a model identifier on huggingface.co/models.")
    parser.add_argument("--dataset_name_or_path", type=str, default="fluers", help="The name of the dataset to use (via the datasets library). It should be either a local path or a dataset identifier on hub.")
    parser.add_argument("--config", type=str, default=None, help="Defines the configuration of the dataset. Default to None.")
    parser.add_argument("--split", type=str, default="test", help="The dataset split to evaluate.")
    parser.add_argument("--num_samples", type=int, default=None, help="The number of samples to evaluate. Default to None.")
    parser.add_argument("--audio_column", type=str, default="audio", help="The name of the audio column in the dataset.")
    parser.add_argument("--text_column", type=str, default="transcription", help="The name of the text column in the dataset.")
    parser.add_argument("--sample_rate", type=int, default=16_000, help="The sample rate to resample the audio to.")
    parser.add_argument("--do_normalize", action="store_true", help="Whether to normalize the text before computing the .")
    parser.add_argument("--seed", type=int, default=42, help="The seed to set for reproducibility.")
    parser.add_argument("--device", type=str, default="auto", help="The device to use for evaluation.")
    parser.add_argument("--num_proc", type=int, default=1, help="The number of processes to use for evaluation.")
    parser.add_argument("--output_dir", type=str, default=None, help="The output directory where the evaluation results will be saved.")

    args = parser.parse_args()
    main(args)

    