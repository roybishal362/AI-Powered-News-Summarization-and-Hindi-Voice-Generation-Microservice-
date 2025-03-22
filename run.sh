#!/bin/bash

# Start the Flask API in the background
python api.py &

# Start the Streamlit app
streamlit run app.py --server.port=$PORT --server.address=$HOST

chmod +x run.sh