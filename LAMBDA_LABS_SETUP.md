# Lambda Labs GPU Cloud Setup Guide

This guide will help you set up and run your adversarial training experiments on Lambda Labs GPU cloud instances.

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
   - **Instance Type**: 1x A100 (40 GB SXM4) - $1.29/hour
   - **Region**: Arizona, USA (or any available)
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
# Example: ssh ubuntu@123.456.789.012
```

### Step 5: Clone Repository
```bash
# Clone your repository
git clone https://github.com/krishnanj/adversarial_attack_gfm.git
cd adversarial_attack_gfm
```

### Step 6: Run Setup Script
```bash
# Make setup script executable and run it
chmod +x scripts/setup_lambda_labs.sh
./scripts/setup_lambda_labs.sh
```

### Step 7: Run Your Experiment
```bash
# Quick experiment (5 iterations, fast parameters)
python scripts/run_lambda_labs_experiment.py --mode quick
```

### Step 8: Monitor Progress (Optional)
```bash
# In another terminal (optional)
./monitor_experiment.sh
```

### Step 9: Download Results
```bash
# On your LOCAL machine (exit SSH first: type 'exit')
# Then run these commands on your laptop:
scp -r ubuntu@<your-instance-ip>:~/adversarial_attack_gfm/results/ ./results/
scp -r ubuntu@<your-instance-ip>:~/adversarial_attack_gfm/plots/ ./plots/
```

### Step 10: Terminate Instance
- **IMPORTANT**: Go back to Lambda Labs dashboard
- Find your instance and click "Terminate"
- **This stops billing!**

## SSH Workflow & Terminal Usage

### Using Your Current Terminal
- **SSH from current terminal**: No need to create new Cursor instances
- **Workflow**: SSH in → Run experiment → Exit SSH → Download results
- **Commands**:
  ```bash
  # Connect to instance
  ssh ubuntu@<instance-ip>
  
  # Run your experiment (you're now ON the Lambda Labs instance)
  git clone https://github.com/krishnanj/adversarial_attack_gfm.git
  cd adversarial_attack_gfm
  ./scripts/setup_lambda_labs.sh
  python scripts/run_lambda_labs_experiment.py --mode quick
  
  # Exit SSH (return to your laptop)
  exit
  
  # Download results (back on your laptop)
  scp -r ubuntu@<instance-ip>:~/adversarial_attack_gfm/results/ ./results/
  ```

### Important Notes
- **SSH connection**: You're "inside" the Lambda Labs instance
- **Exit SSH**: Type `exit` to return to your local machine
- **Download results**: Use `scp` from your local machine (not while SSH'd in)

## Monitoring Your Experiment

### Real-time Monitoring
```bash
# Monitor GPU usage and experiment progress
./monitor_experiment.sh
```

### Check Logs
```bash
# View experiment logs
tail -f lambda_labs_experiment.log

# View training logs
tail -f adversarial_training.log
```

## Configuration Options

### Quick Experiment (Fast Testing)
- **File**: `configs/adversarial_training.yaml`
- **Parameters**:
  - `max_iterations: 5`
  - `train_samples: 1000`
  - `eval_samples: 50`
  - `num_epochs: 1`
  - `test_samples: 5` (for attacks)

### Full Experiment (Complete Research)
- **File**: `configs/adversarial_training.yaml` (modify parameters as needed)
- **To run full experiment**: Modify the config file to increase:
  - `max_iterations: 10` (or more)
  - `train_samples: 47356` (full dataset)
  - `eval_samples: 1000`
  - `num_epochs: 3`
  - `test_samples: 100` (for attacks)

## Cost Management

### Estimated Costs & Timeline
- **Instance Boot**: 2-3 minutes (no charge)
- **Setup Time**: 10-15 minutes
- **Quick Experiment**: 30-60 minutes
- **Total Runtime**: ~45-80 minutes
- **Cost**: ~$1-2 (A100 at $1.29/hour)

### Cost Breakdown
- **A100 (40GB)**: $1.29/hour
- **Quick Experiment**: ~$1-2 total
- **Full Experiment**: ~$3-5 total (if you modify config later)

### Cost Optimization Tips
1. **Use Spot Instances**: 50-70% cheaper than on-demand
2. **Monitor Usage**: Use `nvidia-smi` to check GPU utilization
3. **Terminate Promptly**: Always terminate instances when done
4. **Use Appropriate Instance**: Don't over-provision

### Billing Information
- **Billing Cycle**: Every Friday
- **Credits**: Visible in Settings > Billing
- **Usage Tracking**: Real-time usage in dashboard

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```bash
# Check GPU availability
nvidia-smi

# If not available, check CUDA installation
nvcc --version
```

#### 2. Out of Memory Errors
```bash
# Reduce batch size in config
batch_size: 16  # Instead of 32

# Or use gradient accumulation
gradient_accumulation_steps: 2
```

#### 3. Dataset Not Found
```bash
# Check if datasets are downloaded
ls -la data/raw/GUE/prom/prom_300_all/

# Download datasets if missing
# (Add dataset download commands here)
```

#### 4. Permission Errors
```bash
# Fix file permissions
chmod +x scripts/*.sh
chmod +x scripts/*.py
```

### Getting Help
- **Lambda Labs Support**: [https://lambda.ai/support](https://lambda.ai/support)
- **Documentation**: [https://docs.lambda.ai/](https://docs.lambda.ai/)
- **Community**: Lambda Labs Discord/Forum

## File Structure After Setup

```
adversarial_attack_gfm/
├── scripts/
│   ├── setup_lambda_labs.sh          # Initial setup script
│   ├── run_lambda_labs_experiment.py # Main experiment runner
│   ├── start_experiment.sh           # Quick start script
│   └── monitor_experiment.sh         # Monitoring script
├── configs/
│   ├── adversarial_training.yaml     # Main experiment config (modify for full experiment)
│   └── attack_genetic.yaml          # Attack config
├── results/
│   └── lambda_labs_experiment/       # Experiment results
├── logs/
│   ├── lambda_labs_experiment.log    # Main experiment log
│   └── adversarial_training.log      # Training log
└── LAMBDA_LABS_SETUP.md              # This guide
```

## Experiment Workflow

### Phase 1: Quick Test (Recommended First)
1. Launch small GPU instance (RTX 4090 or V100)
2. Run quick experiment: `python scripts/run_lambda_labs_experiment.py --mode quick`
3. Verify everything works correctly
4. Check results and plots
5. Terminate instance

### Phase 2: Full Experiment
1. Launch larger GPU instance (A100 recommended)
2. Run full experiment: `python scripts/run_lambda_labs_experiment.py --mode full`
3. Monitor progress with `./monitor_experiment.sh`
4. Wait for completion (4-8 hours)
5. Download results and terminate instance

### Phase 3: Analysis
1. Download results from `results/lambda_labs_experiment/`
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
