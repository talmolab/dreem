#!/bin/bash

# DREEM Installation Script
# This script helps users install DREEM with the appropriate dependencies for their platform

set -e

echo "🚀 DREEM Installation Script"
echo "=============================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uv is installed"

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

echo "🔍 Detected platform: $OS $ARCH"

# Determine installation type
if [[ "$OS" == "Darwin" ]]; then
    if [[ "$ARCH" == "arm64" ]]; then
        INSTALL_TYPE="apple-silicon"
        echo "🍎 Installing for Apple Silicon (M1/M2/M3)"
    else
        INSTALL_TYPE="apple-silicon"
        echo "🍎 Installing for Intel Mac"
    fi
elif [[ "$OS" == "Linux" ]] || [[ "$OS" == "CYGWIN"* ]] || [[ "$OS" == "MINGW"* ]]; then
    # Ask user for CUDA preference
    echo "🐧 Linux/Windows detected. Do you want CUDA support? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        INSTALL_TYPE="cuda"
        echo "🚀 Installing with CUDA support"
    else
        INSTALL_TYPE="cpu"
        echo "💻 Installing CPU-only version"
    fi
else
    echo "❌ Unsupported platform: $OS"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
uv sync --extra "$INSTALL_TYPE"

# Ask if user wants dev dependencies
echo "🔧 Do you want to install development dependencies? (y/n)"
read -r dev_response
if [[ "$dev_response" =~ ^[Yy]$ ]]; then
    echo "📦 Installing development dependencies..."
    uv sync --extra dev --extra "$INSTALL_TYPE"
fi

echo "✅ Installation complete!"
echo ""
echo "🎯 To activate the environment, run:"
echo "   source .venv/bin/activate"
echo ""
echo "🎯 Or use uv run to execute commands directly:"
echo "   uv run dreem-train --help"
echo ""
echo "🎯 To run DREEM commands:"
echo "   dreem-train --help"
echo "   dreem-track --help"
echo "   dreem-eval --help"
echo "   dreem-visualize --help"
