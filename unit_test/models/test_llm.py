import sys
sys.path.append('../')
from lits.lm.bedrock_chat import BedrockChatModel

model = BedrockChatModel.load_from_bedrock(
    model_name="anthropic.claude-3-5-sonnet-20240620-v1:0",
    # aws_profile="rmit",
    # aws_region="ap-southeast-2",
    sys_prompt="You are a helpful assistant."
)

out = model("Explain why tree search improves LLM reasoning.")
print(out.text)
