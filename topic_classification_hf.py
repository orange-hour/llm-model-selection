"""
AG News Topic Classification using HuggingFace models locally
"""
import argparse
import time
import pandas as pd
import json
import os
from typing import List
import numpy as np
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
from datetime import datetime


def load_hf_model(model_name: str, device: str = None, task_type: str = "auto"):
    """
    Load a HuggingFace model for topic classification

    Args:
        model_name: HuggingFace model identifier
        device: Device to run on ("cuda", "cpu", or None for auto-detect)
        task_type: Type of model ("auto", "causal-lm", "classification")

    Returns:
        tuple: (pipeline, task_type)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_name}")
    print(f"Using device: {device}")

    # Auto-detect task type if not specified
    if task_type == "auto":
        # Check if it's a classification model
        if any(name in model_name.lower() for name in ["classification", "agnews", "distilbert", "roberta", "bert"]):
            task_type = "classification"
        else:
            task_type = "causal-lm"
        print(f"Auto-detected task type: {task_type}")

    try:
        if task_type == "classification":
            # Use text classification pipeline
            pipe = pipeline(
                "text-classification",
                model=model_name,
                device=0 if device == "cuda" else -1,
                use_fast=False,  # Use slow tokenizer to avoid tiktoken issues
            )
            print(f"✓ Model loaded successfully as classification model")
            return pipe, "classification"
        else:
            # Use text generation pipeline for causal LMs
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            if device == "cuda":
                # When using accelerate with device_map="auto", we must NOT
                # also pass an explicit `device` argument to the pipeline.
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )

                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    # No explicit `device` here; accelerate handles placement.
                )
            else:
                # CPU-only path: do not use accelerate's device_map, and
                # explicitly run the pipeline on CPU.
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map=None,
                )

                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,
                )

            print(f"✓ Model loaded successfully as causal LM")
            return pipe, "causal-lm"
    except Exception as e:
        print(f"Error loading model with task type '{task_type}': {e}")
        raise


def predict_topic(pipe, text: str, task_type: str) -> tuple:
    """
    Predict topic for a single news article using HuggingFace model

    Args:
        pipe: HuggingFace pipeline
        text: News article text (title + description)
        task_type: Type of model ("causal-lm" or "classification")

    Returns:
        tuple: (predicted_topic, raw_model_output)
    """
    try:
        if task_type == "classification":
            # Use classification pipeline directly
            result = pipe(text, truncation=True, max_length=512)
            label_raw = result[0]['label']
            raw_output = str(result[0])  # Store full result for debugging

            # Assume model uses 1-4 indexing (AG News standard)
            # Simply return the label if it's already 1-4
            if label_raw in ['1', '2', '3', '4']:
                return label_raw, raw_output

            return "0", raw_output  # Invalid prediction

        else:
            # Use text generation with prompt for causal LMs
            prompt = f"""Classify the following news article into one of these four topics:

1: World
2: Sports
3: Business
4: Sci/Tech

Read the article carefully and respond with ONLY the number (1, 2, 3, or 4) corresponding to the topic.

Article: {text}

Topic number:"""

            response = pipe(
                prompt,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                truncation=True,
            )

            generated_text = response[0]['generated_text']
            # Extract text after the prompt
            prediction = generated_text[len(prompt):].strip()
            raw_output = prediction  # Store raw prediction for debugging

            # Extract just the number
            for char in prediction:
                if char in ['1', '2', '3', '4']:
                    return char, raw_output

            return "0", raw_output  # Invalid prediction

    except Exception as e:
        print(f"Error predicting topic: {e}")
        return "0", f"ERROR: {str(e)}"


def batch_predict_topics(pipe, texts: List[str], task_type: str, batch_size: int = 8) -> tuple:
    """
    Predict topics for a batch of texts

    Args:
        pipe: HuggingFace pipeline
        texts: List of news article texts
        task_type: Type of model ("causal-lm" or "classification")
        batch_size: Number of samples to process at once

    Returns:
        tuple: (List of predicted topic numbers, List of raw model outputs)
    """
    predictions = []
    raw_outputs = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        batch_results = [predict_topic(pipe, text, task_type) for text in batch]

        # Unpack predictions and raw outputs
        for pred, raw in batch_results:
            predictions.append(pred)
            raw_outputs.append(raw)

        if (i + batch_size) % 100 == 0 or (i + batch_size) >= total:
            print(f"Processed {min(i + batch_size, total)}/{total} samples")

    return predictions, raw_outputs


def topic_classification(
    dataset_path: str = "datasets/agnews_test.jsonl",
    model_name: str = "distilbert-base-uncased",
    device: str = "auto",
    batch_size: int = 8,
    sample_size: int = None,
    save_predictions_path: str = None,
    predictions_format: str = "csv",
    task_type: str = "auto",
) -> pd.DataFrame:
    """
    Run topic classification on AG News dataset using HuggingFace models

    Args:
        dataset_path: Path to the JSONL dataset
        model_name: HuggingFace model identifier
        device: Device to run on ("cuda", "cpu", or "auto")
        batch_size: Number of samples to process at once
        sample_size: Number of samples to classify (None = all)
        save_predictions_path: Directory to save predictions
        predictions_format: Format for saving predictions (csv or json)
        task_type: Type of model ("auto", "causal-lm", "classification")

    Returns:
        DataFrame with results
    """
    print(f"\n{'='*50}")
    print(f"AG News Topic Classification with HuggingFace")
    print(f"{'='*50}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}\n")

    # Load model
    pipe, task_type = load_hf_model(model_name, device, task_type)

    # Load JSONL data
    print(f"\nLoading dataset from {dataset_path}")
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples")

    # Add original row index
    df['original_row_index'] = df.index

    # Sample the data if requested
    if sample_size is not None and sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Randomly sampled {sample_size} rows")
    else:
        df_sample = df.reset_index(drop=True)
        print(f"Using all {len(df_sample)} rows")

    # Combine title and description for better context
    texts = [
        f"{row['title']}. {row['description']}"
        for _, row in df_sample.iterrows()
    ]

    print(f"\nStarting topic classification...")
    print(f"Batch size: {batch_size}")
    print()

    # Run predictions
    start_time = time.time()
    predictions, raw_outputs = batch_predict_topics(pipe, texts, task_type, batch_size)
    elapsed_time = time.time() - start_time

    # Add predictions and raw outputs to dataframe
    df_sample['predicted_topic'] = predictions
    df_sample['raw_model_output'] = raw_outputs

    # Calculate accuracy
    df_sample['ground_truth_topic'] = df_sample['label'].astype(str)
    correct = (df_sample['predicted_topic'] == df_sample['ground_truth_topic']).sum()
    total = len(df_sample)
    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"Topic Classification Results")
    print(f"{'='*50}")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"{'='*50}\n")

    # Topic breakdown
    topic_names = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
    print("\nTopic Distribution:")
    for topic_num in [1, 2, 3, 4]:
        topic_str = str(topic_num)
        gt_count = (df_sample['ground_truth_topic'] == topic_str).sum()
        pred_count = (df_sample['predicted_topic'] == topic_str).sum()
        topic_correct = ((df_sample['ground_truth_topic'] == topic_str) &
                        (df_sample['predicted_topic'] == topic_str)).sum()
        topic_acc = topic_correct / gt_count if gt_count > 0 else 0
        print(f"  {topic_num}: {topic_names[topic_num]:12} - Ground truth: {gt_count:4}, "
              f"Predicted: {pred_count:4}, Accuracy: {topic_acc:.2%}")

    # Save predictions if requested
    if save_predictions_path:
        os.makedirs(save_predictions_path, exist_ok=True)

        # Clean model name for filename
        clean_model_name = model_name.replace(':', '_').replace('/', '_').replace('.', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare output dataframe
        output_df = pd.DataFrame({
            'original_jsonl_row_index': df_sample['original_row_index'],
            'ground_truth_topic': df_sample['ground_truth_topic'],
            'predicted_topic': df_sample['predicted_topic'],
            'raw_model_output': df_sample['raw_model_output'],
            'title': df_sample['title'],
            'description': df_sample['description']
        })

        if predictions_format == "csv":
            output_file = os.path.join(
                save_predictions_path,
                f"topic_predictions_{clean_model_name}.csv"
            )
            output_df.to_csv(output_file, index=False)
            print(f"\n✓ Predictions saved to: {output_file}")
        elif predictions_format == "json":
            output_file = os.path.join(
                save_predictions_path,
                f"topic_predictions_{clean_model_name}.json"
            )
            output_df.to_json(output_file, orient='records', indent=2)
            print(f"\n✓ Predictions saved to: {output_file}")

    return df_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AG News Topic Classification with HuggingFace")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="datasets/agnews_test.jsonl",
        help="Path to AG News JSONL dataset"
    )
    parser.add_argument(
        "-m", "--model-name",
        type=str,
        default="distilbert-base-uncased",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing"
    )
    parser.add_argument(
        "-s", "--sample-size",
        type=int,
        default=None,
        help="Number of samples to classify (None = all)"
    )
    parser.add_argument(
        "--save-predictions",
        type=str,
        default=None,
        help="Directory to save predictions"
    )
    parser.add_argument(
        "--predictions-format",
        type=str,
        default="csv",
        choices=["csv", "json"],
        help="Format for saving predictions"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="auto",
        choices=["auto", "classification", "causal-lm"],
        help="Type of model to use"
    )

    args = parser.parse_args()

    result_df = topic_classification(
        dataset_path=args.dataset,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        save_predictions_path=args.save_predictions,
        predictions_format=args.predictions_format,
        task_type=args.task_type,
    )
