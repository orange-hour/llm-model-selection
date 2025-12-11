"""
Natural Language Inference (NLI) Classification using Ollama
Predicts entailment, neutral, or contradiction relationships
"""

import pandas as pd
import argparse
import asyncio
import aiohttp
import os
import time
from typing import List, Optional
from datetime import datetime


async def predict_nli_ollama(
    session: aiohttp.ClientSession,
    premise: str,
    hypothesis: str,
    model_name: str,
    base_url: str,
) -> tuple:
    """
    Predict NLI relationship using Ollama model.

    Args:
        session: aiohttp session
        premise: Premise text
        hypothesis: Hypothesis text
        model_name: Name of the Ollama model
        base_url: Base URL for Ollama API

    Returns:
        tuple: (predicted_label, raw_model_output)
    """
    prompt = f"""You are tasked with performing Natural Language Inference (NLI). Given a premise and a hypothesis, determine the logical relationship between them.

**Task**: Classify the relationship between the premise and hypothesis as one of the following:
- **entailment** (label: 0): The hypothesis is definitely true given the premise
- **neutral** (label: 1): The hypothesis might be true given the premise, but cannot be confirmed or denied
- **contradiction** (label: 2): The hypothesis is definitely false given the premise

**Input**:
Premise: {premise}
Hypothesis: {hypothesis}

**Instructions**:
1. Carefully read both the premise and hypothesis
2. Determine whether the hypothesis logically follows from the premise (entailment), contradicts the premise (contradiction), or neither (neutral)
3. Respond with only one of these three labels: entailment, neutral, or contradiction

Label:"""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 20,
        }
    }

    try:
        async with session.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status != 200:
                print(f"Error: API returned status {response.status}")
                return "unknown"

            result = await response.json()
            raw_output = result.get("response", "").strip()
            prediction = raw_output.lower()

            # Extract label from response
            if "entailment" in prediction:
                return "entailment", raw_output
            elif "neutral" in prediction:
                return "neutral", raw_output
            elif "contradiction" in prediction:
                return "contradiction", raw_output

            return "unknown", raw_output

    except Exception as e:
        print(f"Error predicting NLI: {e}")
        return "unknown", f"ERROR: {str(e)}"


async def batch_predict_nli(
    premises: List[str],
    hypotheses: List[str],
    model_name: str,
    base_url: str,
    batch_size: int = 10,
) -> tuple:
    """
    Predict NLI relationships for a batch of premise-hypothesis pairs.

    Args:
        premises: List of premise texts
        hypotheses: List of hypothesis texts
        model_name: Name of the Ollama model
        base_url: Base URL for Ollama API
        batch_size: Number of concurrent requests

    Returns:
        tuple: (List of predicted labels, List of raw model outputs)
    """
    predictions = []
    raw_outputs = []

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(premises), batch_size):
            batch_premises = premises[i:i + batch_size]
            batch_hypotheses = hypotheses[i:i + batch_size]

            tasks = [
                predict_nli_ollama(session, premise, hypothesis, model_name, base_url)
                for premise, hypothesis in zip(batch_premises, batch_hypotheses)
            ]
            batch_results = await asyncio.gather(*tasks)

            # Unpack predictions and raw outputs
            for pred, raw in batch_results:
                predictions.append(pred)
                raw_outputs.append(raw)

            if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(premises):
                print(f"Processed {min(i + batch_size, len(premises))}/{len(premises)} samples")

    return predictions, raw_outputs


def nli_classification(
    dataset_path: str = "datasets/snli_test.parquet",
    model_name: str = "llama3.1:70b",
    port: int = 8000,
    sample_size: int = None,
    save_predictions_path: str = None,
    predictions_format: str = "csv",
) -> pd.DataFrame:
    """
    Run NLI classification on SNLI dataset.

    Args:
        dataset_path: Path to the parquet dataset
        model_name: Name of the Ollama model
        port: Port where Ollama server is running
        sample_size: Number of samples to classify (None = all)
        save_predictions_path: Directory to save predictions
        predictions_format: Format for saving predictions (csv or json)

    Returns:
        DataFrame with results
    """
    print(f"\nLoading dataset from {dataset_path}")

    # Load parquet data
    df = pd.read_parquet(dataset_path)
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

    # Extract premises and hypotheses
    premises = df_sample['premise'].tolist()
    hypotheses = df_sample['hypothesis'].tolist()

    print(f"\nUsing model: {model_name}")
    print(f"Ollama API: http://localhost:{port}")
    print(f"Starting NLI classification...")
    print()

    # Run predictions
    base_url = f"http://localhost:{port}"
    start_time = time.time()
    predictions, raw_outputs = asyncio.run(batch_predict_nli(premises, hypotheses, model_name, base_url))
    elapsed_time = time.time() - start_time

    # Add predictions and raw outputs to dataframe
    df_sample['predicted_label'] = predictions
    df_sample['raw_model_output'] = raw_outputs

    # Map label names to numbers for comparison
    label_to_num = {"entailment": 0, "neutral": 1, "contradiction": 2}
    num_to_label = {0: "entailment", 1: "neutral", 2: "contradiction"}

    # Convert ground truth labels to label names
    df_sample['ground_truth_label'] = df_sample['label'].map(num_to_label)

    # Calculate accuracy
    correct = (df_sample['predicted_label'] == df_sample['ground_truth_label']).sum()
    total = len(df_sample)
    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"NLI Classification Results")
    print(f"{'='*50}")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"{'='*50}\n")

    # Label breakdown
    print("\nLabel Distribution:")
    for label_name in ["entailment", "neutral", "contradiction"]:
        gt_count = (df_sample['ground_truth_label'] == label_name).sum()
        pred_count = (df_sample['predicted_label'] == label_name).sum()
        label_correct = ((df_sample['ground_truth_label'] == label_name) &
                        (df_sample['predicted_label'] == label_name)).sum()
        label_acc = label_correct / gt_count if gt_count > 0 else 0
        print(f"  {label_name:14} - Ground truth: {gt_count:4}, "
              f"Predicted: {pred_count:4}, Accuracy: {label_acc:.2%}")

    # Save predictions if requested
    if save_predictions_path:
        os.makedirs(save_predictions_path, exist_ok=True)

        # Clean model name for filename
        clean_model_name = model_name.replace(':', '_').replace('/', '_').replace('.', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare output dataframe
        output_df = pd.DataFrame({
            'original_parquet_row_index': df_sample['original_row_index'],
            'ground_truth_label': df_sample['ground_truth_label'],
            'predicted_label': df_sample['predicted_label'],
            'raw_model_output': df_sample['raw_model_output'],
            'premise': df_sample['premise'],
            'hypothesis': df_sample['hypothesis']
        })

        if predictions_format == "csv":
            output_file = os.path.join(
                save_predictions_path,
                f"nli_predictions_{clean_model_name}.csv"
            )
            output_df.to_csv(output_file, index=False)
            print(f"\n✓ Predictions saved to: {output_file}")
        elif predictions_format == "json":
            output_file = os.path.join(
                save_predictions_path,
                f"nli_predictions_{clean_model_name}.json"
            )
            output_df.to_json(output_file, orient='records', indent=2)
            print(f"\n✓ Predictions saved to: {output_file}")

    return df_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNLI NLI Classification with Ollama")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="datasets/snli_test.parquet",
        help="Path to SNLI parquet dataset"
    )
    parser.add_argument(
        "-m", "--model-name",
        type=str,
        default="llama3.1:70b",
        help="Ollama model name"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8000,
        help="Port where Ollama server is running"
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

    args = parser.parse_args()

    result_df = nli_classification(
        dataset_path=args.dataset,
        model_name=args.model_name,
        port=args.port,
        sample_size=args.sample_size,
        save_predictions_path=args.save_predictions,
        predictions_format=args.predictions_format,
    )
