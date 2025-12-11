"""
AG News Topic Classification using Ollama
Predicts news topics (World, Sports, Business, Sci/Tech) from text
"""

import pandas as pd
import json
import argparse
import asyncio
import aiohttp
import os
import time
from typing import List, Optional
from datetime import datetime


async def predict_topic_ollama(
    session: aiohttp.ClientSession,
    text: str,
    model_name: str,
    base_url: str,
) -> str:
    """
    Predict news topic using Ollama model.

    Args:
        session: aiohttp session
        text: News text (title + description)
        model_name: Name of the Ollama model
        base_url: Base URL for Ollama API

    Returns:
        Predicted topic number (1-4) as string
    """
    prompt = """Classify the following news article into one of these four topics:

1: World
2: Sports
3: Business
4: Sci/Tech

Read the article carefully and respond with ONLY the number (1, 2, 3, or 4) corresponding to the topic.

Article: {text}

Topic number:""".format(text=text)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 10,
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
                return "0"

            result = await response.json()
            prediction = result.get("response", "").strip()

            # Extract just the number
            for char in prediction:
                if char in ['1', '2', '3', '4']:
                    return char

            return "0"  # Invalid prediction

    except Exception as e:
        print(f"Error predicting topic: {e}")
        return "0"


async def batch_predict_topics(
    texts: List[str],
    model_name: str,
    base_url: str,
    batch_size: int = 10,
) -> List[str]:
    """
    Predict topics for a batch of texts.

    Args:
        texts: List of news texts
        model_name: Name of the Ollama model
        base_url: Base URL for Ollama API
        batch_size: Number of concurrent requests

    Returns:
        List of predicted topic numbers
    """
    predictions = []

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tasks = [
                predict_topic_ollama(session, text, model_name, base_url)
                for text in batch
            ]
            batch_predictions = await asyncio.gather(*tasks)
            predictions.extend(batch_predictions)

            if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(texts):
                print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} samples")

    return predictions


def topic_classification(
    dataset_path: str = "datasets/agnews_test.jsonl",
    model_name: str = "llama3.1:70b",
    port: int = 8000,
    sample_size: int = 385,
    save_predictions_path: str = None,
    predictions_format: str = "csv",
) -> pd.DataFrame:
    """
    Run topic classification on AG News dataset.

    Args:
        dataset_path: Path to the JSONL dataset
        model_name: Name of the Ollama model
        port: Port where Ollama server is running
        sample_size: Number of samples to classify
        save_predictions_path: Directory to save predictions
        predictions_format: Format for saving predictions (csv or json)

    Returns:
        DataFrame with results
    """
    print(f"\nLoading dataset from {dataset_path}")

    # Load JSONL data
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples")

    # Add original row index before sampling
    df['original_row_index'] = df.index

    # Sample the data
    if sample_size < len(df):
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

    print(f"\nUsing model: {model_name}")
    print(f"Ollama API: http://localhost:{port}")
    print(f"Starting topic classification...")
    print()

    # Run predictions
    base_url = f"http://localhost:{port}"
    start_time = time.time()
    predictions = asyncio.run(batch_predict_topics(texts, model_name, base_url))
    elapsed_time = time.time() - start_time

    # Add predictions to dataframe
    df_sample['predicted_topic'] = predictions

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
        print(f"  {topic_num}: {topic_names[topic_num]:12} - Ground truth: {gt_count:3}, "
              f"Predicted: {pred_count:3}, Accuracy: {topic_acc:.2%}")

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
    parser = argparse.ArgumentParser(description="AG News Topic Classification with Ollama")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="datasets/agnews_test.jsonl",
        help="Path to AG News JSONL dataset"
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
        default=385,
        help="Number of samples to classify"
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

    result_df = topic_classification(
        dataset_path=args.dataset,
        model_name=args.model_name,
        port=args.port,
        sample_size=args.sample_size,
        save_predictions_path=args.save_predictions,
        predictions_format=args.predictions_format,
    )
