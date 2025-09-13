#!/bin/bash
# Lambda Labs GPU Cloud Setup Script
# This script sets up the environment for running adversarial training on Lambda Labs

set -e

echo "🚀 Setting up Lambda Labs GPU Cloud Environment..."

# Check if we're on Lambda Labs
if [[ -z "$LAMBDA_LABS" ]]; then
    echo "⚠️  This script is designed for Lambda Labs environment"
    echo "   Set LAMBDA_LABS=1 if running on Lambda Labs"
fi

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y git curl wget

# Install Python 3.9 if not available
echo "🐍 Setting up Python 3.9..."
sudo apt-get install -y python3.9 python3.9-venv python3.9-dev python3-pip

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3.9 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (for GPU instances)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
echo "📚 Installing project dependencies..."
pip install -r requirements.txt

# Install additional GPU monitoring tools
echo "📊 Installing GPU monitoring tools..."
pip install gpustat nvidia-ml-py3

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/raw data/processed data/adversarial
mkdir -p models/baseline models/adversarial_training
mkdir -p plots results logs

# Set up Git (if not already configured)
echo "🔧 Setting up Git..."
if [[ -z "$(git config --global user.name)" ]]; then
    git config --global user.name "Lambda Labs User"
    git config --global user.email "user@lambda.ai"
fi

# Clone or update repository
if [[ ! -d ".git" ]]; then
    echo "📥 Cloning repository..."
    # Replace with your actual repository URL
    git clone https://github.com/krishnanj/adversarial_attack_gfm.git .
fi

# Download datasets (if not present)
echo "📊 Setting up datasets..."
if [[ ! -f "data/raw/GUE/prom/prom_300_all/train.csv" ]]; then
    echo "   Downloading GUE promoter dataset..."
    # Add dataset download logic here
    mkdir -p data/raw/GUE/prom/prom_300_all
    # You'll need to add the actual dataset download commands
fi

# Set environment variables
echo "🌍 Setting environment variables..."
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Create a startup script for easy activation
echo "📝 Creating startup script..."
cat > start_experiment.sh << 'EOF'
#!/bin/bash
# Quick start script for Lambda Labs experiments

echo "🚀 Starting Adversarial Training Experiment on Lambda Labs..."

# Activate virtual environment
source venv/bin/activate

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi
gpustat

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Run the experiment
echo "🏃 Running adversarial training experiment..."
python scripts/run_adversarial_training.py

echo "✅ Experiment completed!"
echo "📊 Results saved to: results/adversarial_training/"
echo "📈 Plots saved to: plots/adversarial_training/"
EOF

chmod +x start_experiment.sh

# Create monitoring script
echo "📊 Creating monitoring script..."
cat > monitor_experiment.sh << 'EOF'
#!/bin/bash
# Monitor experiment progress

echo "📊 Monitoring experiment progress..."

while true; do
    clear
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    
    echo -e "\n=== Process Status ==="
    ps aux | grep python | grep -v grep
    
    echo -e "\n=== Disk Usage ==="
    df -h /tmp /home
    
    echo -e "\n=== Recent Logs ==="
    tail -n 10 adversarial_training.log 2>/dev/null || echo "No log file found"
    
    sleep 30
done
EOF

chmod +x monitor_experiment.sh

echo "✅ Lambda Labs setup completed!"
echo ""
echo "🚀 To start your experiment:"
echo "   ./start_experiment.sh"
echo ""
echo "📊 To monitor progress:"
echo "   ./monitor_experiment.sh"
echo ""
echo "💡 Remember to terminate your instance when done!"
echo "   https://cloud.lambda.ai/login"
