#!/bin/bash

# Install Docker Sandbox Dependencies
echo "🐳 Installing Docker sandbox dependencies..."

# Install Docker Python SDK
pip install docker

echo "✅ Docker sandbox dependencies installed successfully!"

# Check if Docker is installed and running
if command -v docker &> /dev/null; then
    if docker info &> /dev/null; then
        echo "✅ Docker is installed and running"
    else
        echo "⚠️  Docker is installed but not running. Please start Docker daemon:"
        echo "   sudo systemctl start docker"
        echo "   or start Docker Desktop if using macOS/Windows"
    fi
else
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   - Linux: https://docs.docker.com/engine/install/"
    echo "   - macOS: https://docs.docker.com/desktop/install/mac-install/"
    echo "   - Windows: https://docs.docker.com/desktop/install/windows-install/"
fi

echo ""
echo "🚀 Docker sandbox is now ready to use!"
echo "   The sandbox will automatically build Docker images as needed."
echo "   To switch back to Firejail, change USE_DOCKER_SANDBOX=False in execute_code.py"
