# Author : Abdul Waheed

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torchaudio
from tqdm import tqdm
import functools
import jiwer
import json 
import librosa

from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Initialize normalizer
normalizer = BasicTextNormalizer()

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
    normalized_gt = [normalizer(gt) for gt in ground_truths]
    normalized_hypothesis = [normalizer(h) for h in hypothesis]

    # Compute metrics using jiwer
    word_output = jiwer.process_words(
        reference=normalized_gt,
        hypothesis=normalized_hypothesis
    )
    characters_output = jiwer.process_characters(
        reference=normalized_gt,
        hypothesis=normalized_hypothesis,
    )
    
    # Compute total errors
    nwErrors = word_output.substitutions + word_output.insertions + word_output.deletions
    ncErrors = characters_output.substitutions + characters_output.insertions + characters_output.deletions
    
    WER = word_output.wer
    CER = characters_output.cer
    
    # scaling the WER and CER so that the scale is same as o and ncErrors

    return {
        "WER": round(WER*100, 2),
        "CER": round(CER*100, 2),
        "nwErrors": nwErrors,
        "ncErrors": ncErrors,
        "wRefLength": len(word_output.references[0]),  # Length of the reference
        "cRefLength": len(characters_output.references[0])  # Length of the reference
    }

def load_model(model_name_or_path, device="auto"):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path, device_map=device)
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    return {"model": model, "processor": processor}

def transcribe_audio(model_dict, audio):
    processor = model_dict["processor"]
    model = model_dict["model"]
    inputs = processor(audio, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = model(input_values=inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)


def main(args):
    
    # Set seed for everything - reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Load the dataset and the model
    dataset = load_dataset(args.dataset_name_or_path, args.config, split=args.split)
    
    # sample the dataset
    if args.n_samples is not None:
        dataset = dataset.shuffle(seed=args.seed).select(range(min(args.n_samples, len(dataset))))
    
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
    
    # Compute the metrics
    metrics = compute_metrics(ground_truths=ground_truths, hypothesis=hypothesis)
    
    model_id = args.model_name_or_path.split("/")[-1]
    dataset_id = args.dataset_name_or_path.split("/")[-1]
    metrics["model_id"] = model_id
    metrics["dataset_id"] = dataset_id
    metrics["split"] = args.split
    metrics["n_samples"] = len(dataset)
    
    
    # Print the metrics
    print(metrics)
    
    # Save the metrics
    
    if args.output_dir is not None:
        output_dir = args.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / f"{model_id}_{dataset_id}_{args.split}.json"
        with open(output_path, "w") as f:
            json.dump(metrics, f)
    
    
    
    
    
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate the uDistilWhisper models.')
    
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-large-v2", help="The model checkpoint for weights initialization. It should be either a local path or a model identifier on huggingface.co/models.")
    parser.add_argument("--dataset_name_or_path", type=str, default="fluers", help="The name of the dataset to use (via the datasets library). It should be either a local path or a dataset identifier on hub.")
    parser.add_argument("--config", type=str, default=None, help="Defines the configuration of the dataset. Default to None.")
    parser.add_argument("--split", type=str, default="test", help="The dataset split to evaluate.")
    parser.add_argument("--n_samples", type=int, default=None, help="The number of samples to evaluate. Default to None.")
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

    