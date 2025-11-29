#!/usr/bin/env bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
