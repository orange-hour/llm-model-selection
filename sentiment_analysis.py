from src.core.udfs import llm_naive
from src.core.utils import clean_df, convert_df, prepend_col_name
import argparse
import time
import pandas as pd
import os
import asyncio
from pyspark.sql import functions as F

from src.core.utils import prepend_col_name
from pyspark.sql.functions import col, lit
from typing import List
import numpy as np
from pathlib import Path
from pyspark.sql.functions import upper, col

import concurrent.futures

simulate_runtime = 0


async def call_LLM_UDF(aggregated_df, cols, prompt_prefix: str, guided_choice: List[str] = None, port: int = 8000):
    # If guided choice is None, set it to positive/negative for sentiment analysis
    if guided_choice is None:
        guided_choice = ["positive", "negative"]

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        llm_s = time.time()
        result_df = await loop.run_in_executor(
            pool,
            lambda: aggregated_df.select(
                llm_naive(lit(prompt_prefix), col("contexts"), lit(cols), lit(guided_choice), lit(port)).alias("results")
            ),
        )
        result_exploded = await loop.run_in_executor(pool, lambda: result_df.withColumn("results", F.explode("results")).cache())
        output_length = await loop.run_in_executor(pool, lambda: result_exploded.count())
        llm_e = time.time()
        llm_time = llm_e - llm_s

        return llm_time, output_length, result_exploded


async def sentiment_analysis(
    dataset_path: str = "datasets/IMDB_reviews.csv",
    output_path: str = None,
    guided_choice: List[str] = None,
    num_gpus: int = 1,
    assigned_port: int = 8000,
    save_predictions_path: str = None,
    predictions_format: str = "csv",
    model_name: str = "unknown",
):
    sql_operator_time = 0
    algo_runtime = 0

    sql_read_s = time.time()
    # Prompt for sentiment analysis
    prompt_prefix = "Analyze the sentiment of the following review. Respond with either 'positive' or 'negative'."

    algo = "naive"
    sql_operator_time = time.time() - sql_read_s

    sql_s = time.time()
    # Read the IMDB dataset
    full_dataset_path = os.path.join(Path(__file__).resolve().parent.parent.parent, dataset_path)
    df_original = pd.read_csv(full_dataset_path)

    # Keep review and sentiment columns (sentiment is ground truth)
    if 'review' not in df_original.columns or 'sentiment' not in df_original.columns:
        raise ValueError("Dataset must contain both 'review' and 'sentiment' columns")

    # Add original row index before sampling (0-based index from CSV)
    df_original['original_row_index'] = df_original.index

    # Randomly sample 385 rows from the dataset
    sample_size = 385
    if len(df_original) > sample_size:
        df_original = df_original.sample(n=sample_size, random_state=42)
        print(f"Randomly sampled {sample_size} rows from the dataset")
        print(f"Original row indices (from CSV): {sorted(df_original['original_row_index'].tolist())[:10]}... (showing first 10)")
    else:
        print(f"Dataset has {len(df_original)} rows, which is less than {sample_size}. Using all rows.")

    # Store ground truth labels and original indices before processing
    ground_truth_df = df_original[['sentiment', 'original_row_index']].copy()
    ground_truth_df['row_id'] = range(len(ground_truth_df))

    # Keep only the review column for LLM processing
    df = df_original[['review']].copy()
    df = clean_df(df)
    df = prepend_col_name(df)

    num_total = df.shape[0]
    sql_operator_time += time.time() - sql_s

    print(f"Dataset Characteristics: number of rows: {num_total}, number of columns: {df.shape[1]}")
    print(f"Column Orders: {df.columns}")

    algo_s = time.time()
    # No reordering for naive algorithm - shuffle the sampled data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    algo_e = time.time()
    algo_runtime += algo_e - algo_s
    print(f"Algorithm runtime: {algo_runtime:.3f}", flush=True)

    # Run LLMs, and actually collect the data
    sql_s = time.time()
    cols = list(df.columns)
    df_partitions = np.array_split(df, num_gpus)  # Split DF into num_gpus partitions
    partition_aggregates = []
    for _, df_partition in enumerate(df_partitions):
        df_new = convert_df(df_partition)
        sample_df_with_contexts = df_new.withColumn("context", F.struct(cols))
        # Group the entire dataframe into one row, and collect contexts into a list
        aggregated_df = sample_df_with_contexts.groupBy().agg(F.collect_list(col("context")).alias("contexts"))
        partition_aggregates.append(aggregated_df)
    sql_e = time.time()
    sql_operator_time += sql_e - sql_s

    print(f"Number of GPUs: {num_gpus}")

    end_to_end_llm_s = time.time()
    llm_port_to_time = {}
    llm_port_to_output_len = {}
    starting_port = 8000
    result_dfs = {}

    async def execute_llm_task(i, df_partition):
        if num_gpus == 1 and assigned_port:
            print(f"Use assigned port: {assigned_port}", flush=True)
            port = assigned_port
        else:
            # Just assign sequentially
            port = starting_port + i * 100

        # Executing LLM
        llm_time, output_length, result_exploded = await call_LLM_UDF(df_partition, cols, prompt_prefix, guided_choice, port)
        print(f"Port {port} finished, LLM time: {llm_time:.3f}, output length: {output_length}")
        llm_port_to_time[port] = llm_time
        llm_port_to_output_len[port] = output_length
        result_dfs[port] = result_exploded

    # Create a list of tasks and run them concurrently
    tasks = [execute_llm_task(i, partition_aggregates[i]) for i in range(num_gpus)]
    await asyncio.gather(*tasks)

    assert sum(llm_port_to_output_len.values()) == num_total

    end_to_end_llm_e = time.time()
    llm_time = max(llm_port_to_time.values())
    actual_end_to_end = end_to_end_llm_e - end_to_end_llm_s

    print(f"LLM port to time: {llm_port_to_time}")
    print(f"Actual end-to-end LLM time: {actual_end_to_end:.3f}, take max of LLM times: {llm_time:.3f}")

    sql_start_s = time.time()
    result_dfs = list(result_dfs.values())
    if result_dfs:
        result_df_combined = result_dfs[0]
        if len(result_dfs) > 1:
            for df in result_dfs[1:]:
                result_df_combined = result_df_combined.union(df)

        # Count positive and negative sentiments
        positive_count = result_df_combined.filter(upper(col("results")) == "POSITIVE").count()
        negative_count = result_df_combined.filter(upper(col("results")) == "NEGATIVE").count()

        # Convert predictions to pandas for accuracy calculation
        predictions_pandas = result_df_combined.toPandas()
        predictions_pandas['predicted_sentiment'] = predictions_pandas['results'].str.lower()
        predictions_pandas['row_id'] = range(len(predictions_pandas))

        # Merge with ground truth
        comparison_df = ground_truth_df.merge(predictions_pandas[['row_id', 'predicted_sentiment']], on='row_id')
        comparison_df['ground_truth_lower'] = comparison_df['sentiment'].str.lower()

        # Calculate accuracy
        correct_predictions = (comparison_df['ground_truth_lower'] == comparison_df['predicted_sentiment']).sum()
        total_predictions = len(comparison_df)
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0

    else:
        result_df_combined = None
        positive_count = 0
        negative_count = 0
        accuracy = 0
        correct_predictions = 0
        total_predictions = 0

    sql_operator_time += time.time() - sql_start_s

    rps = num_total / llm_time
    tot_time = sql_operator_time + algo_runtime + llm_time

    # Optionally persist predictions to disk
    if save_predictions_path:
        try:
            fmt = (predictions_format or "csv").lower()
            if result_df_combined is None:
                print("No predictions to save.")
            else:
                if fmt == "csv":
                    # Use comparison_df which has all the info: ground truth, predictions, original indices
                    save_df = comparison_df[['original_row_index', 'sentiment', 'predicted_sentiment']].copy()
                    save_df.columns = ['original_csv_row_index', 'ground_truth_sentiment', 'predicted_sentiment']

                    # Create output directory if it doesn't exist
                    os.makedirs(save_predictions_path, exist_ok=True)

                    # Clean model name for filename (replace special characters)
                    clean_model_name = model_name.replace(':', '_').replace('/', '_').replace(' ', '_')
                    output_file = os.path.join(save_predictions_path, f'predictions_{clean_model_name}.csv')

                    save_df.to_csv(output_file, index=False)
                    print(f"Saved predictions with original row indices to {output_file}")
                    print(f"Columns: original_csv_row_index, ground_truth_sentiment, predicted_sentiment")
                elif fmt == "parquet":
                    save_df = comparison_df[['original_row_index', 'sentiment', 'predicted_sentiment']].copy()
                    save_df.columns = ['original_csv_row_index', 'ground_truth_sentiment', 'predicted_sentiment']

                    # Convert to Spark DataFrame and save
                    from pyspark.sql import SparkSession
                    spark = SparkSession.builder.getOrCreate()
                    spark_df = spark.createDataFrame(save_df)
                    spark_df.write.mode("overwrite").parquet(save_predictions_path)
                    print(f"Saved predictions as Parquet to {save_predictions_path}")
                else:
                    print(f"Unknown predictions format: {predictions_format}. Supported: csv, parquet")
        except Exception as e:
            print(f"Error while saving predictions: {e}")

    # Output statistics
    if output_path:
        with open(output_path, "w") as f:
            print(f"Algorithm: {algo}", file=f)
            print(f"Number of rows: {num_total}", file=f)
            print(f"Algorithm Runtime: {algo_runtime}", file=f)
            print(f"LLM time: {llm_time}", file=f)
            print(f"SQL Operators time: {sql_operator_time}", file=f)
            print(f"Total time: {tot_time}", file=f)
            print(f"Requests per Second (RPS): {rps}", file=f)
            print(f"Positive sentiments: {positive_count}", file=f)
            print(f"Negative sentiments: {negative_count}", file=f)
            print(f"Correct predictions: {correct_predictions}/{total_predictions}", file=f)
            print(f"Accuracy: {accuracy:.2f}%", file=f)
    else:
        print("*" * 25 + "Result" + "*" * 25)
        print(f"Algorithm: {algo}")
        print(f"Number of rows: {num_total}")
        print(f"Algorithm Runtime: {algo_runtime}")
        print(f"LLM time: {llm_time}")
        print(f"SQL Operators time: {sql_operator_time}")
        print(f"Total time: {tot_time}")
        print(f"Requests per Second (RPS): {rps}")
        print(f"Positive sentiments: {positive_count}")
        print(f"Negative sentiments: {negative_count}")
        print(f"Correct predictions: {correct_predictions}/{total_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")

    return result_df_combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_path",
        default="datasets/IMDB_reviews.csv",
        help="Path to IMDB reviews dataset",
    )
    parser.add_argument("--output_path", help="Saves statistics output to a file")

    # Query specific arguments
    parser.add_argument(
        "-g",
        "--guided_choice",
        nargs="+",
        help="Constrained decoding output options (default: positive, negative)",
    )

    # Distributed execution
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("-p", "--port", type=int, default=None, help="Assigned port (num_gpu = 1)")
    parser.add_argument(
        "--save-predictions",
        help="Path or directory to save full predictions (CSV directory or Parquet file/directory)",
        default=None,
    )
    parser.add_argument(
        "--predictions-format",
        choices=["csv", "parquet"],
        default="csv",
        help="Format to save predictions when --save-predictions is provided",
    )
    parser.add_argument(
        "--model-name",
        default="unknown",
        help="Name of the model being used (for output filename)",
    )

    args = parser.parse_args()
    guided_choice = [str(x) for x in args.guided_choice] if args.guided_choice else ["positive", "negative"]

    # Run the query end-to-end
    result_df = asyncio.run(
        sentiment_analysis(
            dataset_path=args.dataset_path,
            output_path=args.output_path,
            guided_choice=guided_choice,
            num_gpus=args.num_gpus,
            assigned_port=args.port,
            save_predictions_path=args.save_predictions,
            predictions_format=args.predictions_format,
            model_name=args.model_name,
        )
    )
