# AWS Bedrock Inference Profiles - Q&A

## What are the two ways to access models in AWS Bedrock?

AWS Bedrock provides two methods to access models:

1. **Direct Model ID** (region-specific)
   - Format: `anthropic.claude-3-5-haiku-20241022-v1:0`
   - Only works in specific regions where the model is deployed
   - May require provisioned throughput

2. **Cross-Region Inference Profile** (region-agnostic)
   - Format: `us.anthropic.claude-3-5-haiku-20241022-v1:0`
   - Works from any region within the profile's geography
   - Allows on-demand access without provisioned throughput

## What does the `us.` prefix mean?

The `us.` prefix is an **inference profile identifier**, not a region code:

- `us.` = US-based inference profile (routes to US regions where model is available)
- `eu.` = EU-based inference profile (routes to EU regions where model is available)

It tells AWS Bedrock which geographic area to route your request to, but you can call it from any region.

## Does the inference profile override my `region_name` parameter?

**No.** The `region_name` in `session.client("bedrock-runtime", region_name=region)` is NOT overridden.

Here's what happens:

```python
# You connect to us-east-1
client = session.client("bedrock-runtime", region_name="us-east-1")

# You use a cross-region inference profile
model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"

# Flow:
# 1. Your API call goes to us-east-1 Bedrock endpoint
# 2. Bedrock sees the "us." inference profile
# 3. Bedrock routes internally to wherever Claude Haiku is available in US regions
# 4. Response comes back through us-east-1
```

The inference profile is about **where the model runs**, not where you call from.

## Where is the inference profile stored?

The inference profile is **managed by AWS Bedrock**, not stored in your code. It's a routing mechanism:

- When you use `us.anthropic.*`, AWS routes to US regions
- When you use `eu.anthropic.*`, AWS routes to EU regions
- When you use `anthropic.*` (no prefix), it only works in your specified region

## Is using `us.anthropic.claude-3-5-haiku-20241022-v1:0` the same as using `anthropic.claude-3-5-haiku-20241022-v1:0` with the correct region?

**Mostly yes, but with important differences.**

### Region Flexibility

**With Inference Profile:**
```python
# Works from ANY US region
client = session.client("bedrock-runtime", region_name="us-west-2")
model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"  # ✓ Works

client = session.client("bedrock-runtime", region_name="us-east-1")
model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"  # ✓ Works
```

**With Direct Model ID:**
```python
# Only works in regions where model is deployed
client = session.client("bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"  # ✓ Works (if deployed there)

client = session.client("bedrock-runtime", region_name="us-west-2")
model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"  # ✗ May fail (if not deployed there)
```

### Key Differences

| Aspect | Inference Profile | Direct Model ID |
|--------|------------------|-----------------|
| Region flexibility | Works from any region in profile geography | Must be in exact deployment region |
| Routing | AWS handles automatically | No routing - must exist in your region |
| Resilience | AWS manages regional failover | Manual region switching required |
| Setup | Don't need to know model locations | Need to track which regions have model |

## Why did my direct model ID fail with "requires provisioned throughput"?

When you use a direct model ID like `anthropic.claude-3-5-haiku-20241022-v1:0`, AWS may require you to have provisioned throughput (pre-purchased capacity) for that model in that specific region.

**Error example:**
```
ValidationException: Invocation of model ID anthropic.claude-3-5-haiku-20241022-v1:0 
with on-demand throughput isn't supported. Retry your request with the ID or ARN of 
an inference profile that contains this model.
```

**Solution:** Use the inference profile instead:
```python
# Instead of this:
model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"

# Use this:
model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
```

## What are the benefits of using inference profiles?

1. **Flexibility** - Works from any region within the profile's geography
2. **Simplicity** - Don't need to track which regions have which models
3. **Resilience** - AWS handles routing if a region has issues
4. **On-demand access** - No need for provisioned throughput
5. **Future-proof** - AWS can add/remove regions transparently

## Example: Complete workflow

```python
import boto3

# Connect to Bedrock from any US region
session = boto3.Session()
client = session.client("bedrock-runtime", region_name="us-west-2")

# Use inference profile for flexibility
response = client.converse(
    modelId="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    messages=[
        {
            "role": "user",
            "content": [{"text": "Hello, how are you?"}]
        }
    ],
    inferenceConfig={
        "maxTokens": 2048,
        "temperature": 0.7
    }
)

print(response['output']['message']['content'][0]['text'])
```

## How does this relate to the test code?

In `test_load_models.py`, we use:

```python
BEDROCK_MODEL = "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0"
```

The `bedrock/` prefix is our internal convention to indicate it's a Bedrock model. The actual model ID passed to AWS is `us.anthropic.claude-3-5-haiku-20241022-v1:0`.

This allows the tests to work from any AWS region without needing to know where Claude Haiku is specifically deployed.

## Quick Reference

| Model ID Format | Example | Use Case |
|----------------|---------|----------|
| Direct | `anthropic.claude-3-5-haiku-20241022-v1:0` | When you know exact region and have provisioned throughput |
| US Profile | `us.anthropic.claude-3-5-haiku-20241022-v1:0` | On-demand access from any US region |
| EU Profile | `eu.anthropic.claude-3-5-haiku-20241022-v1:0` | On-demand access from any EU region |

**Recommendation:** Use inference profiles (`us.*` or `eu.*`) for most use cases unless you have specific requirements for direct model IDs.


## What's the difference between "inference profile" and AWS config "profile"?

These are **completely different concepts** that unfortunately share the same word "profile":

### 1. Bedrock Inference Profile (Model Routing)

**What it is:** A model identifier prefix that tells Bedrock where to route your API request.

**Format:** `us.anthropic.claude-3-5-haiku-20241022-v1:0`

**Purpose:** Routes your model inference request to appropriate AWS regions

**Managed by:** AWS Bedrock service

**Used in:** Model ID parameter in API calls

```python
# This is a Bedrock inference profile
response = client.converse(
    modelId="us.anthropic.claude-3-5-haiku-20241022-v1:0",  # ← Inference profile
    messages=[...]
)
```

### 2. AWS Config Profile (Authentication/Configuration)

**What it is:** A named set of AWS credentials and configuration settings stored locally.

**Format:** `[profile rmit-sso]` in `~/.aws/config`

**Purpose:** Stores authentication credentials, default region, and other AWS CLI/SDK settings

**Managed by:** You (stored in `~/.aws/config` and `~/.aws/credentials`)

**Used in:** Authenticating and configuring AWS SDK/CLI

```python
# This is an AWS config profile
session = boto3.Session(profile_name="rmit-sso")  # ← AWS config profile
client = session.client("bedrock-runtime")
```

### Summary Table

| Aspect | Bedrock Inference Profile | AWS Config Profile |
|--------|--------------------------|-------------------|
| Purpose | Model routing | Authentication & config |
| Location | Part of model ID | `~/.aws/config` file |
| Example | `us.anthropic.claude-*` | `[profile rmit-sso]` |
| Managed by | AWS Bedrock | You (local files) |
| Used for | Which region runs model | Which credentials to use |

### They Work Together

```python
# Step 1: Use AWS config profile to authenticate
session = boto3.Session(profile_name="rmit-sso")  # ← AWS config profile
client = session.client("bedrock-runtime", region_name="us-east-1")

# Step 2: Use Bedrock inference profile to call model
response = client.converse(
    modelId="us.anthropic.claude-3-5-haiku-20241022-v1:0",  # ← Bedrock inference profile
    messages=[{"role": "user", "content": [{"text": "Hello"}]}]
)
```

## What does the content in `~/.aws/config` mean?

Let's break down your config file:

```ini
[default]
region = us-east-1
sso_start_url = https://rmit-research.awsapps.com/start
sso_region = ap-southeast-2
sso_account_id = 554674964376
sso_role_name = RMIT-ResearchAdmin

[profile rmit-sso]
sso_session = rmit
sso_account_id = 554674964376
sso_role_name = RMIT-ResearchAdmin
region = us-east-1

[sso-session rmit]
sso_start_url = https://rmit-research.awsapps.com/start
sso_region = ap-southeast-2
sso_registration_scopes = sso:account:access
```

### Section 1: `[default]` Profile

```ini
[default]
region = us-east-1                                          # Default AWS region for API calls
sso_start_url = https://rmit-research.awsapps.com/start    # SSO login portal URL
sso_region = ap-southeast-2                                 # Region where SSO service is hosted
sso_account_id = 554674964376                               # AWS account ID to access
sso_role_name = RMIT-ResearchAdmin                          # IAM role to assume after login
```

**What it means:**
- This is your default profile (used when you don't specify a profile name)
- Uses AWS SSO (Single Sign-On) for authentication
- After SSO login, assumes the `RMIT-ResearchAdmin` role in account `554674964376`
- API calls default to `us-east-1` region

**Usage:**
```python
# Uses [default] profile automatically
session = boto3.Session()
```

### Section 2: `[profile rmit-sso]` Profile

```ini
[profile rmit-sso]
sso_session = rmit                    # References the [sso-session rmit] section
sso_account_id = 554674964376         # AWS account ID
sso_role_name = RMIT-ResearchAdmin    # IAM role to assume
region = us-east-1                    # Default region
```

**What it means:**
- Named profile called `rmit-sso`
- Uses the shared SSO session configuration named `rmit`
- Same account and role as default, but explicitly named

**Usage:**
```python
# Explicitly use rmit-sso profile
session = boto3.Session(profile_name="rmit-sso")
```

### Section 3: `[sso-session rmit]` Session

```ini
[sso-session rmit]
sso_start_url = https://rmit-research.awsapps.com/start    # SSO portal URL
sso_region = ap-southeast-2                                 # SSO service region
sso_registration_scopes = sso:account:access                # Permission scopes
```

**What it means:**
- Shared SSO session configuration named `rmit`
- Can be referenced by multiple profiles
- Defines where to authenticate (RMIT's SSO portal in ap-southeast-2)
- `sso:account:access` scope allows accessing AWS accounts

**Why separate?**
- Multiple profiles can share the same SSO session
- You only need to login once for all profiles using this session

### How SSO Authentication Works

```
1. You run: aws sso login --profile rmit-sso
   ↓
2. Opens browser to: https://rmit-research.awsapps.com/start
   ↓
3. You login with RMIT credentials
   ↓
4. SSO service (in ap-southeast-2) authenticates you
   ↓
5. You get temporary credentials for account 554674964376
   ↓
6. Credentials are cached locally (~/.aws/sso/cache/)
   ↓
7. Your code uses these credentials to call AWS APIs
```

### Complete Example

```python
import boto3

# Method 1: Use default profile
session = boto3.Session()  # Uses [default]
client = session.client("bedrock-runtime", region_name="us-east-1")

# Method 2: Use named profile
session = boto3.Session(profile_name="rmit-sso")  # Uses [profile rmit-sso]
client = session.client("bedrock-runtime", region_name="us-east-1")

# Both authenticate via SSO session [sso-session rmit]
# Both assume role RMIT-ResearchAdmin in account 554674964376
# Both default to us-east-1 region

# Now use Bedrock inference profile (different concept!)
response = client.converse(
    modelId="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    messages=[{"role": "user", "content": [{"text": "Hello"}]}]
)
```

### Key Concepts Summary

| Config Element | Purpose | Example |
|---------------|---------|---------|
| `[default]` | Default profile when none specified | Auto-used by boto3 |
| `[profile name]` | Named profile you can select | `profile_name="rmit-sso"` |
| `[sso-session name]` | Shared SSO configuration | Referenced by profiles |
| `region` | Default AWS region for API calls | `us-east-1` |
| `sso_start_url` | Where to login | RMIT SSO portal |
| `sso_region` | Where SSO service runs | `ap-southeast-2` |
| `sso_account_id` | Which AWS account to access | `554674964376` |
| `sso_role_name` | Which IAM role to assume | `RMIT-ResearchAdmin` |

### Common Confusion Clarified

```python
# AWS Config Profile (authentication)
session = boto3.Session(profile_name="rmit-sso")  # ← From ~/.aws/config
client = session.client("bedrock-runtime", region_name="us-east-1")

# Bedrock Inference Profile (model routing)
response = client.converse(
    modelId="us.anthropic.claude-3-5-haiku-20241022-v1:0",  # ← From Bedrock service
    messages=[...]
)
```

**Remember:**
- **AWS config profile** = Who you are (authentication)
- **Bedrock inference profile** = Where the model runs (routing)
- They're completely independent concepts!
