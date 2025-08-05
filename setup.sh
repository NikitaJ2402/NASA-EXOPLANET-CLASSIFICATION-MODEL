#!/bin/bash
# Setup script for NASA Exoplanet Classifier Streamlit App

echo "🌌 Setting up NASA Exoplanet Classifier..."

# Install required packages
echo "📦 Installing required packages..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "🚀 To run the app:"
echo "   streamlit run streamlit_exoplanet_app.py"
echo ""
echo "🌐 The app will open automatically in your browser at http://localhost:8501"
