#!/bin/bash

# Setup script for VC Due Diligence Tool
# This script sets up the environment and installs all dependencies

set -e  # Exit on error

echo "======================================"
echo "VC Due Diligence Tool - Setup"
echo "======================================"
echo ""

# Check if Xcode Command Line Tools are installed
if ! xcode-select -p &> /dev/null; then
    echo "❌ Xcode Command Line Tools not found"
    echo ""
    echo "Please install Xcode Command Line Tools first:"
    echo "  xcode-select --install"
    echo ""
    echo "After installation completes, run this script again."
    exit 1
else
    echo "✅ Xcode Command Line Tools installed"
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    echo ""
    echo "Please install Python 3.11 or later:"
    echo "  brew install python@3.11"
    exit 1
else
    PYTHON_VERSION=$(python3 --version)
    echo "✅ Python installed: $PYTHON_VERSION"
fi

# Check if we're in the correct directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "(This may take a few minutes)"
echo ""
pip install -r requirements.txt

echo ""
echo "✅ Dependencies installed successfully"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  .env file not found"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your API keys:"
    echo "   - ANTHROPIC_API_KEY (required)"
    echo "   - OPENAI_API_KEY (required)"
    echo ""
else
    echo "✅ .env file exists"
fi

# Initialize database
echo ""
echo "Initializing database..."
python tools/init_database.py

echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Add your API keys to .env file:"
echo "   - ANTHROPIC_API_KEY"
echo "   - OPENAI_API_KEY"
echo ""
echo "3. Start the API server:"
echo "   python app/main.py"
echo "   # Or use: uvicorn app.main:app --reload"
echo ""
echo "4. Test with sample CLI:"
echo "   python test_cli.py"
echo ""
echo "5. Open API documentation:"
echo "   http://localhost:8000/docs"
echo ""
echo "======================================"
