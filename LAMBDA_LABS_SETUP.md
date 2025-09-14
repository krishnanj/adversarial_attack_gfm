# Lambda Labs GPU Cloud Setup Guide

This guide will help you set up and run your adversarial training experiments on Lambda Labs GPU cloud instances with support for both promoter and transcription factor datasets.

## Complete Step-by-Step Workflow

### Step 1: Access Lambda Labs Dashboard
1. Go to [https://cloud.lambda.ai/login](https://cloud.lambda.ai/login)
2. Sign up/Login with your account
3. You should see your cloud credits in the dashboard

### Step 2: Set Up SSH Key (One-time setup)
1. **On your local machine**, generate SSH key:
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   # Press Enter for default location (~/.ssh/id_rsa)
   # Press Enter twice for no passphrase
   ```

2. **Copy your public key**:
   ```bash
   cat ~/.ssh/id_rsa.pub
   ```

3. **Add to Lambda Labs**:
   - In Lambda Labs dashboard, go to "SSH Keys" section
   - Click "Add SSH Key"
   - Paste your public key
   - Give it a name (e.g., "my-laptop")

### Step 3: Launch GPU Instance
1. **In Lambda Labs dashboard**:
   - Click "Create Instance"
   - **Instance Type**: 1x H100 SXM5 (80 GB) - $2.89/hour (recommended for full experiments)
   - **Region**: us-south-3 (or any available)
   - **Base Image**: Lambda Stack 22.04
   - **Filesystem**: Don't attach a filesystem
   - **Security**: No rulesets
   - **SSH Key**: Select your SSH key
   - Click "Launch Instance"

2. **Wait for instance to start** (2-3 minutes)
3. **Note the IP address** when ready

### Step 4: Connect to Instance
```bash
# SSH into your instance (use your current terminal)
ssh ubuntu@<your-instance-ip>
# Example: ssh ubuntu@192.222.54.196
```

### Step 5: Clone Repository
```bash
# Clone your repository
git clone https://github.com/krishnanj/adversarial_attack_gfm.git
cd adversarial_attack_gfm
```

### Step 6: Set Up Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 7: Fix DNABERT-2 Compatibility (CRITICAL)
This is the most important step - DNABERT-2 requires a specific Triton setup:

```bash
# Install build dependencies
pip install cmake

# Clone and install Triton from source
git clone https://github.com/openai/triton.git
cd triton
pip install -e .

# Remove Triton (this is the key step!)
pip uninstall triton

# Go back to your project directory
cd ~/adversarial_attack_gfm
```

**Why this works**: DNABERT-2 needs Triton to be importable but then falls back to standard PyTorch attention, avoiding the `trans_b` compatibility error.

### Step 8: Run Your Experiment

#### Option A: Transcription Factor Dataset (Recommended)
```bash
# Run transcription factor dataset with full training (5 iterations, ~2-3 hours)
nohup python scripts/run_adversarial_training.py --dataset tf > training_output.log 2>&1 &
```

#### Option B: Promoter Dataset
```bash
# Run promoter dataset with full training
nohup python scripts/run_adversarial_training.py --dataset promoter > training_output.log 2>&1 &
```

#### Option C: Quick Testing (Fast)
```bash
# For quick testing, modify configs first:
# Set train_samples: 500, val_samples: 100, num_epochs: 1, test_samples: 5
# Then run:
nohup python scripts/run_adversarial_training.py --dataset tf > training_output.log 2>&1 &
```

### Step 9: Monitor Progress
```bash
# Check if the process is running
ps aux | grep python

# Monitor the log file in real-time
tail -f training_output.log

# Check GPU usage
nvidia-smi

# Check disk space
df -h
```

### Step 10: Download Results
```bash
# On your LOCAL machine (exit SSH first: type 'exit')
# Then run these commands on your laptop:
scp -r ubuntu@<your-instance-ip>:~/adversarial_attack_gfm/results/ ./results/
scp -r ubuntu@<your-instance-ip>:~/adversarial_attack_gfm/plots/ ./plots/
scp ubuntu@<your-instance-ip>:~/adversarial_attack_gfm/training_output.log ./training_output.log
```

### Step 11: Terminate Instance
- **IMPORTANT**: Go back to Lambda Labs dashboard
- Find your instance and click "Terminate"
- **This stops billing!**

## Dataset Options

### Transcription Factor Dataset (Recommended)
- **Path**: `data/raw/GUE/tf/0/`
- **Sequences**: 32,378 training samples
- **Length**: 200bp sequences
- **Command**: `python scripts/run_adversarial_training.py --dataset tf`

### Promoter Dataset
- **Path**: `data/raw/GUE/prom/prom_300_all/`
- **Sequences**: 47,356 training samples
- **Length**: 300bp sequences
- **Command**: `python scripts/run_adversarial_training.py --dataset promoter`

## Configuration Options

### Full Training (Current Default)
- **File**: `configs/adversarial_training.yaml`
- **Parameters**:
  - `max_iterations: 5`
  - `num_epochs: 3`
  - `batch_size: 16`
  - `warmup_steps: 500`
  - `convergence_threshold: 0` (no early stopping)
- **Runtime**: ~2-3 hours
- **Cost**: ~$6-9 (H100 at $2.89/hour)

### Quick Testing
To run quick tests, modify `configs/adversarial_training.yaml`:
```yaml
training:
  train_samples: 500
  val_samples: 100
  num_epochs: 1
  warmup_steps: 5
  batch_size: 128

# And configs/attack_genetic.yaml:
attack:
  test_samples: 5
genetic_algorithm:
  population_size: 10
  max_generations: 3
```
- **Runtime**: ~2-3 minutes
- **Cost**: ~$0.10-0.15

## Background Execution Commands

### Start Experiment in Background
```bash
# Transcription factor dataset
nohup python scripts/run_adversarial_training.py --dataset tf > training_output.log 2>&1 &

# Promoter dataset
nohup python scripts/run_adversarial_training.py --dataset promoter > training_output.log 2>&1 &
```

### Monitor Background Process
```bash
# Check if running
ps aux | grep python

# View logs
tail -f training_output.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Stop Background Process (if needed)
```bash
# Find process ID
ps aux | grep python

# Kill process
kill <process_id>
```

## Expected Results

### Training Progress
- **Iteration 1**: Initial training + attack generation
- **Iteration 2-5**: Retraining with adversarial examples
- **Final**: Model robustness evaluation

### Output Files
- **Results**: `results/adversarial_training/`
- **Plots**: `plots/adversarial_training/`
- **Logs**: `training_output.log`
- **Metrics**: `data/adversarial/iterative_training/training_metrics.csv`

### Key Metrics
- **Baseline Accuracy**: Initial model performance
- **Final Accuracy**: After adversarial training
- **Attack Success Rate**: Percentage of successful attacks
- **Perturbation Efficiency**: Average changes needed for attacks

## Cost Management

### Estimated Costs & Timeline
- **Instance Boot**: 2-3 minutes (no charge)
- **Setup Time**: 10-15 minutes
- **Full Experiment**: 2-3 hours
- **Total Runtime**: ~2.5-3.5 hours
- **Cost**: ~$7-10 (H100 at $2.89/hour)

### Cost Optimization Tips
1. **Use Spot Instances**: 50-70% cheaper than on-demand
2. **Monitor Usage**: Use `nvidia-smi` to check GPU utilization
3. **Terminate Promptly**: Always terminate instances when done
4. **Use Appropriate Instance**: H100 for full experiments, smaller for testing

## Troubleshooting

### Common Issues

#### 1. Triton Compatibility Error
```
TypeError: dot() got an unexpected keyword argument 'trans_b'
```
**Solution**: Follow Step 7 exactly - install Triton from source then remove it.

#### 2. GPU Not Detected
```bash
# Check GPU availability
nvidia-smi

# If not available, check CUDA installation
nvcc --version
```

#### 3. Out of Memory Errors
```bash
# Reduce batch size in config
batch_size: 8  # Instead of 16

# Or use gradient accumulation
gradient_accumulation_steps: 2
```

#### 4. Dataset Not Found
```bash
# Check if datasets are downloaded
ls -la data/raw/GUE/tf/0/
ls -la data/raw/GUE/prom/prom_300_all/

# Datasets should be included in the repository
```

#### 5. Early Stopping (Fixed)
The config now has `convergence_threshold: 0` which disables early stopping and ensures all 5 iterations run.

### Getting Help
- **Lambda Labs Support**: [https://lambda.ai/support](https://lambda.ai/support)
- **Documentation**: [https://docs.lambda.ai/](https://docs.lambda.ai/)
- **Community**: Lambda Labs Discord/Forum

## File Structure After Setup

```
adversarial_attack_gfm/
├── scripts/
│   ├── run_adversarial_training.py    # Main experiment runner (supports both datasets)
│   ├── run_genetic_attack.py          # Standalone attack script
│   ├── plot_results.py                # Plot generation
│   ├── extract_attack_stats.py        # Statistics extraction
│   └── generate_plots.py              # Plot generation
├── configs/
│   ├── adversarial_training.yaml      # Main experiment config
│   └── attack_genetic.yaml           # Attack config
├── data/
│   ├── raw/GUE/tf/0/                 # Transcription factor dataset
│   └── raw/GUE/prom/prom_300_all/    # Promoter dataset
├── results/
│   └── adversarial_training/          # Experiment results
├── plots/
│   └── adversarial_training/          # Generated plots
└── LAMBDA_LABS_SETUP.md              # This guide
```

## Experiment Workflow

### Phase 1: Quick Test (Recommended First)
1. Launch GPU instance (H100 or smaller)
2. Run quick test: Modify configs for fast parameters
3. Run: `nohup python scripts/run_adversarial_training.py --dataset tf > training_output.log 2>&1 &`
4. Verify everything works correctly
5. Check results and plots
6. Terminate instance

### Phase 2: Full Experiment
1. Launch H100 GPU instance
2. Run full experiment: `nohup python scripts/run_adversarial_training.py --dataset tf > training_output.log 2>&1 &`
3. Monitor progress with `tail -f training_output.log`
4. Wait for completion (2-3 hours)
5. Download results and terminate instance

### Phase 3: Analysis
1. Download results from `results/adversarial_training/`
2. Generate additional plots if needed
3. Analyze results for paper preparation

## Security Notes

- **SSH Keys**: Use SSH key authentication
- **Firewall**: Configure security groups appropriately
- **Data**: Don't store sensitive data on instances
- **Termination**: Always terminate instances when done

## Support Contacts

- **Lambda Labs Support**: [https://lambda.ai/support](https://lambda.ai/support)
- **Technical Issues**: Check logs and documentation first
- **Billing Questions**: Check dashboard billing section

---

**Important Reminders:**
1. Always terminate your instances when done to avoid charges
2. Monitor your usage and costs regularly
3. Keep backups of important results
4. Use appropriate instance sizes for your workload
5. **CRITICAL**: Follow Step 7 (Triton fix) exactly to avoid compatibility errors
6. The system now supports both promoter and transcription factor datasets
7. Early stopping is disabled - all 5 iterations will run as requested