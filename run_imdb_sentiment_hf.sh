#!/bin/bash

# Script to run IMDB sentiment analysis with HuggingFace models

# Configuration
MODEL_NAME="meta-llama/Llama-3.2-1B"  # Change this to your desired HF model
DATASET_PATH="datasets/IMDB_reviews.csv"
OUTPUT_DIR="/work/11079/tiffanysh/vista/artifact-eval/output"
DEVICE="gpu"  # Options: auto, cuda, cpu
BATCH_SIZE=8

echo "=================================================="
echo "IMDB Sentiment Analysis with HuggingFace"
echo "=================================================="
echo ""
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Device: $DEVICE"
echo "Batch size: $BATCH_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if Python environment has required packages
echo "Checking dependencies..."
python -c "import transformers; import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Required packages not found"
    echo "Please install: pip install transformers torch"
    exit 1
fi
echo "✓ Dependencies found"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs/imdb_sentiment_hf

# Run sentiment analysis
echo "Starting sentiment analysis..."
echo ""

python -m src.pyspark.sentiment_analysis_hf \
    -d "$DATASET_PATH" \
    -m "$MODEL_NAME" \
    --device "$DEVICE" \
    --batch-size $BATCH_SIZE \
    --save-predictions "$OUTPUT_DIR" \
    --predictions-format csv \
    2>&1 | tee logs/imdb_sentiment_hf/run.log

ANALYSIS_EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    echo "✓ Sentiment analysis completed successfully!"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo "Logs saved to: logs/imdb_sentiment_hf/run.log"
else
    echo "✗ Sentiment analysis failed with exit code $ANALYSIS_EXIT_CODE"
    echo ""
    echo "Check logs at: logs/imdb_sentiment_hf/run.log"
fi
echo "=================================================="

exit $ANALYSIS_EXIT_CODE
