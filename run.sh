#!/bin/bash

# Set default port values if not provided
export PORT=${PORT:-7860}
export HOST=${HOST:-0.0.0.0}
export FLASK_PORT=${FLASK_PORT:-5000}

echo "Starting API server on port $FLASK_PORT..."
python api.py &
API_PID=$!

# Wait for API to initialize
sleep 5
echo "API server started with PID: $API_PID"

echo "Starting Streamlit app on port $PORT..."
streamlit run app.py --server.port=$PORT --server.address=$HOST

# If Streamlit exits, kill the API process
kill $API_PID