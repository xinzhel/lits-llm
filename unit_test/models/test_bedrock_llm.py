import sys
sys.path.append('../../')
from lits.lm import get_lm, BedrockChatModel


model_name= "bedrock/amazon.titan-text-express-v1"

model = get_lm(model_name)

try:
    print(model.get_loglikelihood("I lost my key today. This is ", ["I lost my key today. This is " + "good"])[0])
except Exception as e:
    print(f"Could not compute loglikelihood: {e}")
    print("Note: AWS Bedrock models (including Command R and Titan) do not currently support returning token log-probabilities via the API.")

# print(model("hello, how are you?").text)
# cohere.command-r-v1:0
#  # anthropic.claude-sonnet-4-5-20250929-v1:0
# "bedrock/anthropic.claude-opus-4-5-20251101-v1:0"
# "anthropic.claude-3-5-sonnet-20240620-v1:0",

# return_logprobs": true