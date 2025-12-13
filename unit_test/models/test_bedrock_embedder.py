import boto3
import json

# After running:  aws sso login --profile my-sso
session = boto3.Session()
client = session.client("bedrock-runtime", region_name="us-east-1")

model_id = "cohere.embed-v4:0" # "amazon.titan-embed-text-v2:0"
if "cohere" in model_id:
    response = client.invoke_model(
        modelId= model_id,
        body=json.dumps({
            "texts": ["hello world"],
            "input_type": "search_document"   # or "search_query", "classification", etc.
        })
    )
if "amazon.titan" in model_id:
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "inputText": "hello world",
            # Optional: choose dimension etc. if you need
            # "embeddingConfig": {"outputEmbeddingLength": 1024}
        }),
    )

# response["body"] is a StreamingBody; read and parse JSON
resp_body = json.loads(response["body"].read())
# Titan returns {"embedding": [float, ...]} or {"embeddings": [...]},
# depending on variant; adjust as needed:
embedding = resp_body.get("embedding") or resp_body.get("embeddings")

print(embedding)