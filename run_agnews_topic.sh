#!/bin/bash

# Script to run AG News topic classification with Ollama

# Set the path to ollama (update this to your ollama installation path)
OLLAMA_PATH="/work/11079/tiffanysh/vista/bin/ollama"

MODEL_NAME="llama3.1:70b"
PORT=8000
DATASET_PATH="datasets/agnews_test.jsonl"
OUTPUT_DIR="/work/11079/tiffanysh/vista/artifact-eval/output"
SAMPLE_SIZE=10000

echo "=================================================="
echo "AG News Topic Classification with Ollama"
echo "=================================================="
echo ""

# Step 1: Check if Ollama is installed
echo "[1/5] Checking if Ollama is installed..."
if [ ! -f "$OLLAMA_PATH" ]; then
    echo "ERROR: Ollama is not found at $OLLAMA_PATH"
    echo "Please update OLLAMA_PATH in the script to point to your ollama installation"
    exit 1
fi
echo "✓ Ollama is installed at $OLLAMA_PATH"
echo ""

# Step 2: Check if the model is available
echo "[2/5] Checking if model '$MODEL_NAME' is available..."
if ! $OLLAMA_PATH list | grep -q "$MODEL_NAME"; then
    echo "Model '$MODEL_NAME' not found. Pulling model..."
    $OLLAMA_PATH pull "$MODEL_NAME"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to pull model '$MODEL_NAME'"
        exit 1
    fi
fi
echo "✓ Model '$MODEL_NAME' is available"
echo ""

# Step 3: Kill any existing Ollama server on the port
echo "[3/5] Checking for existing Ollama server on port $PORT..."
PID=$(lsof -t -i:$PORT 2>/dev/null)
if [ ! -z "$PID" ]; then
    echo "Killing existing process $PID on port $PORT"
    kill -9 $PID
    sleep 2
fi
echo "✓ Port $PORT is available"
echo ""

# Step 4: Start Ollama server
echo "[4/5] Starting Ollama server on port $PORT..."
mkdir -p logs/agnews_topic
OLLAMA_HOST=0.0.0.0:$PORT OLLAMA_ORIGINS="*" $OLLAMA_PATH serve > logs/agnews_topic/ollama_server.log 2>&1 &
SERVER_PID=$!
echo "Ollama server started with PID: $SERVER_PID"

# Wait for server to start with health check
echo "Waiting for server to start..."
MAX_WAIT=60
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$PORT/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama server is responding"
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    echo "Waiting... ($ELAPSED/$MAX_WAIT seconds)"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: Ollama server failed to start within $MAX_WAIT seconds"
    echo "Check logs at: logs/agnews_topic/ollama_server.log"
    cat logs/agnews_topic/ollama_server.log
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Preload the model
echo "Preloading model '$MODEL_NAME'..."
OLLAMA_HOST=localhost:$PORT $OLLAMA_PATH run "$MODEL_NAME" "" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "WARNING: Failed to preload model, but continuing anyway"
fi
echo "✓ Ollama server is ready"
echo ""

# Step 5: Run topic classification
echo "[5/5] Running topic classification on AG News dataset..."
echo "Dataset: $DATASET_PATH"
echo "Sample size: $SAMPLE_SIZE rows"
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo ""

# Verify server is still running and accessible
if ! curl -s http://localhost:$PORT/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama server is not responding on port $PORT"
    echo "Server logs:"
    cat logs/agnews_topic/ollama_server.log
    exit 1
fi

echo "Starting topic classification..."
echo "Predictions will be saved to: $OUTPUT_DIR"
python -m src.pyspark.topic_classification \
    -d "$DATASET_PATH" \
    -m "$MODEL_NAME" \
    -p $PORT \
    -s $SAMPLE_SIZE \
    --save-predictions "$OUTPUT_DIR" \
    --predictions-format csv

ANALYSIS_EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    echo "✓ Topic classification completed successfully!"
else
    echo "✗ Topic classification failed with exit code $ANALYSIS_EXIT_CODE"
fi
echo "=================================================="
echo ""

# Cleanup: Stop Ollama server
echo "Stopping Ollama server (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null

# Also kill any remaining processes on the port
PID=$(lsof -t -i:$PORT 2>/dev/null)
if [ ! -z "$PID" ]; then
    kill -9 $PID 2>/dev/null
fi

echo "✓ Cleanup completed"
echo ""

if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    echo "Results have been displayed above."
    echo "Server logs are available at: logs/agnews_topic/ollama_server.log"
fi

exit $ANALYSIS_EXIT_CODE
