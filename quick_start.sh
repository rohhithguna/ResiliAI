#!/bin/bash

echo "🚀 Setting up OpenEnv ResiliAI Project..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Run validation
echo "🧪 Running validation..."
python validate_tasks.py

if [ $? -eq 0 ]; then
    echo "✅ Validation passed! Launching demo..."
    echo "🎬 Starting Streamlit app (press Ctrl+C to stop)..."
    streamlit run frontend/app.py
else
    echo "❌ Validation failed. Please check the output above."
    exit 1
fi
