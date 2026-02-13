# TGI Remote Models

LITS supports remote completion models served via HuggingFace TGI (Text Generation Inference). This is useful for:
- Running large models on GPU instances (EC2, etc.) without local GPU
- Sharing a model server across multiple experiments
- RAP formulation which requires completion models (not chat)

## Quick Start

```python
from lits.lm import get_lm

# Option 1: Direct connection (if EC2 security group allows your IP)
model = get_lm("tgi://YOUR_EC2_PUBLIC_IP:8080/meta-llama/Meta-Llama-3-8B")

# Option 2: Using environment variable
import os
os.environ["TGI_ENDPOINT"] = "http://YOUR_EC2_PUBLIC_IP:8080"
model = get_lm("tgi:///meta-llama/Meta-Llama-3-8B")

# Generate completion
output = model("Question: What is 2+2?\nAnswer:")
print(output.text)
```

## URL Format

```
tgi://host:port/model_name
```

- `host:port`: TGI server address (use `localhost:8080` with SSH tunnel)
- `model_name`: HuggingFace model ID (for logging/identification)

## EC2 Deployment

Use the provided deployment script to launch a TGI server on EC2:

```bash
# For Llama-3-8B (RAP formulation)
export HF_TOKEN=hf_your_token_here
bash aws/deploy_thinkprm/ec2_launch_thinkprm.sh --model llama3

# For ThinkPRM-14B (reward model)
bash aws/deploy_thinkprm/ec2_launch_thinkprm.sh --model thinkprm

# Start SSH tunnel (required for local access)
bash aws/deploy_thinkprm/ec2_launch_thinkprm.sh --model llama3 --tunnel

# Test the endpoint
bash aws/deploy_thinkprm/ec2_launch_thinkprm.sh --model llama3 --test
```

## Using with RAP

RAP (Reasoning via Planning) requires completion models. Use TGI to serve Llama-3-8B remotely:

```bash
python main_search.py \
    --include lits_benchmark.formulations.rap \
    --search_framework rap \
    --dataset gsm8k \
    --policy_model_name "tgi://localhost:8080/meta-llama/Meta-Llama-3-8B" \
    --eval_model_name "tgi://localhost:8080/meta-llama/Meta-Llama-3-8B" \
    --search-arg n_actions=3 \
    --search-arg max_steps=10
```

## TGI Server Setup

If deploying manually (not using the EC2 script):

```bash
# Pull TGI Docker image
docker pull ghcr.io/huggingface/text-generation-inference:2.4.1

# Run TGI with Llama-3-8B
docker run --gpus all -p 8080:80 \
    -e HF_TOKEN=$HF_TOKEN \
    -v ~/.cache/huggingface:/data \
    ghcr.io/huggingface/text-generation-inference:2.4.1 \
    --model-id meta-llama/Meta-Llama-3-8B \
    --max-input-length 4096 \
    --max-total-tokens 8192
```

## API Endpoints

TGI exposes:
- `/generate` - Completion endpoint (used by `TGIModel`)
- `/v1/chat/completions` - OpenAI-compatible chat endpoint
- `/health` - Health check

## Instance Types

| Model | Instance | GPU | VRAM | Cost/hr |
|-------|----------|-----|------|---------|
| Llama-3-8B | g5.2xlarge | 1x A10G | 24GB | ~$1.21 |
| Llama-3-70B | g5.12xlarge | 4x A10G | 96GB | ~$5.67 |
| ThinkPRM-14B | g5.12xlarge | 4x A10G | 96GB | ~$5.67 |

## Troubleshooting

**Connection refused**: Ensure SSH tunnel is running
```bash
bash aws/deploy_thinkprm/ec2_launch_llama3.sh --tunnel
```

**Model not loading**: Check TGI logs on EC2
```bash
ssh -i ~/.ssh/your_key.pem ubuntu@<ip> 'sudo docker logs tgi'
```

**Timeout errors**: Increase timeout in TGIModel
```python
model = get_lm("tgi://localhost:8080/model", timeout=300)
```
