# %%
import json
import logging
logger = logging.getLogger(__name__)
from copy import deepcopy
from datetime import datetime
import pytz
from mem0 import Memory
from mem0.memory.main import _build_filters_and_metadata, get_update_memory_messages
from mem0.memory.utils import (
    extract_json,
    get_fact_retrieval_messages,
    parse_messages,
    parse_vision_messages,
    process_telemetry_filters,
    remove_code_blocks,
)
from mem0.configs.base import MemoryConfig
from mem0.memory.telemetry import capture_event
import concurrent
from mem0.utils.factory import (
    EmbedderFactory,
    GraphStoreFactory,
    LlmFactory,
    VectorStoreFactory,
    RerankerFactory,
)
from mem0.llms.aws_bedrock import AWSBedrockLLM
from mem0.vector_stores.qdrant import Qdrant
from mem0.configs.base import MemoryConfig
from mem0.configs.vector_stores.qdrant import QdrantConfig
from mem0.configs.vector_stores.s3_vectors import S3VectorsConfig

# %%
import boto3
s = boto3.Session()
creds = s.get_credentials().get_frozen_credentials()
print("Access key:", creds.access_key)
print("Session token:", creds.token)


# %%
config = MemoryConfig()
# embedder configuration
config.embedder.provider = "huggingface"
config.embedder.config = {"model": "sentence-transformers/multi-qa-mpnet-base-cos-v1"}

# vector store configuration
config.vector_store.provider = "qdrant"
qdrant_config = QdrantConfig(
    collection_name="mem0",
    embedding_model_dims=1536,
    client=None,
    host=None,
    port=None,
    path="/tmp/qdrant",
    url=None,
    api_key=None,
    on_disk=False
)
config.vector_store.config = qdrant_config

# llm configuration
# config.llm.provider = "vllm"
# config.llm.config = {"model": "Qwen/Qwen3-0.6B", "temperature": 0.2, "max_tokens": 1024}
config.llm.provider = "aws_bedrock"
config.llm.config = {"model": "anthropic.claude-3-5-haiku-20241022-v1:0", "temperature": 0.2, "max_tokens": 2000}

# create memory instance
memory = Memory(config=config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
metadata ={"category": "movie_recommendations"}
# result = memory.add(messages, user_id="alice", metadata=metadata)



# %% [markdown]
# `memory.__init__()`

# %%
from mem0.llms.vllm import VllmLLM
config.graph_store.config

# %% [markdown]
# llm
# 

# %%
config.llm.config = {"model": "NousResearch/Hermes-3-Lla.2-3B", "temperature": 0.2, "max_tokens": 1024}
llm = LlmFactory.create(config.llm.provider, config.llm.config)

# import litellm
# litellm.supports_function_calling(model="huggingface/NousResearch/Hermes-4-405B")

# %% [markdown]
# Create embedding model

# %%
print(config.embedder.provider)
print(config.embedder.config)


# %%
config = MemoryConfig()
self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )

# How to customize the behaviour in the Memory.__init__ to do the following?
# EmbedderFactory.create(provider_name="huggingface",  config=embedder_config, vector_config=config.vector_store.config)

# Here is the solution to replace the default OpenAI embedder with HuggingFace embedder
# config = MemoryConfig()
# config.embedder.provider = "huggingface"
# config.embedder.config = {"model": "sentence-transformers/multi-qa-mpnet-base-cos-v1"}

from mem0.embeddings.openai import OpenAIEmbedding
from mem0.embeddings.huggingface import HuggingFaceEmbedding

# %% [markdown]
# Create vector db

# %%
print(config.vector_store.provider)
print(config.vector_store.config)

# %%
from mem0.vector_stores.qdrant import Qdrant

qdrant_config = QdrantConfig(
    collection_name="mem0",
    embedding_model_dims=1536,
    client=None,
    host=None,
    port=None,
    path="/tmp/qdrant",
    url=None,
    api_key=None,
    on_disk=False
)
config.vector_store.provider = "qdrant"

# %%
print(memory.vector_store)
print(memory.vector_store.client)
print(memory.vector_store.client._client)
print(memory.vector_store.client._client.location) # QdrantLocal

# print(memory.vector_store.client._client._api_key) # QdrantClient
# print(memory.vector_store.client._client._host) # QdrantClient
# print(memory.vector_store.client._client._port) # QdrantClient
# print(memory.vector_store.client._client.rest_uri) # QdrantClient
# print(memory.vector_store.client._client._host) # QdrantClient
# print(memory.vector_store.client._client._port) # QdrantClient

# %% [markdown]
# `memory.add`

# %%
print(config.llm.config.get("enable_vision"))
print(memory.enable_graph)

# %%
processed_metadata, effective_filters = _build_filters_and_metadata(
    user_id="alice",
    agent_id=None,
    run_id=None,
    input_metadata=None,
)
print(processed_metadata)
print(effective_filters)

# %%
messages = parse_vision_messages(messages)
for message in messages:
    print(message)
# messages = [
#     {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
#     {"role": "assistant", "content": "How about thriller movies? They can be quite engaging."},
#     {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
#     {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
# ]

# %%

with concurrent.futures.ThreadPoolExecutor() as executor:
    future1 = executor.submit(memory._add_to_vector_store, messages, processed_metadata, effective_filters, infer=True)
    future2 = [] # executor.submit(memory._add_to_graph, messages, effective_filters) # if memory.enable_graph is True

    concurrent.futures.wait([future1, future2])

    vector_store_result = future1.result()
    graph_result = future2.result()




# %% [markdown]
# `memory._add_to_vector_store`

# %%
print(config.custom_fact_extraction_prompt)
metadata ={"category": "movie_recommendations"}
filters = effective_filters

# %%
parsed_messages = parse_messages(messages)
print(f"Parsed Messages (type: {type(parsed_messages)}):\n{parsed_messages} ")


# %%
if config.custom_fact_extraction_prompt:
    system_prompt = config.custom_fact_extraction_prompt
    user_prompt = f"Input:\n{parsed_messages}"
else:
    # Determine if this should use agent memory extraction based on agent_id presence
    # and role types in messages
    is_agent_memory = memory._should_use_agent_memory_extraction(messages, metadata)
    system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages, is_agent_memory)

response = memory.llm.generate_response(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    response_format={"type": "json_object"},
)
response

# %%
response

# %%

try:
    response = remove_code_blocks(response)
    if not response.strip():
        new_retrieved_facts = []
    else:
        try:
            # First try direct JSON parsing
            new_retrieved_facts = json.loads(response)["facts"]
        except json.JSONDecodeError:
            # Try extracting JSON from response using built-in function
            extracted_json = extract_json(response)
            new_retrieved_facts = json.loads(extracted_json)["facts"]
except Exception as e:
    # logger.error(f"Error in new_retrieved_facts: {e}")
    new_retrieved_facts = []
print(new_retrieved_facts)

# %%

# if not new_retrieved_facts:
#     logger.debug("No new facts retrieved from input. Skipping memory update LLM call.")

retrieved_old_memory = []
new_message_embeddings = {}
# Search for existing memories using the provided session identifiers
# Use all available session identifiers for accurate memory retrieval
search_filters = {}
if filters.get("user_id"):
    search_filters["user_id"] = filters["user_id"]
if filters.get("agent_id"):
    search_filters["agent_id"] = filters["agent_id"]
if filters.get("run_id"):
    search_filters["run_id"] = filters["run_id"]
print(search_filters)


# %%
# for new_mem in new_retrieved_facts:
new_mem = new_retrieved_facts[0]
messages_embeddings = memory.embedding_model.embed(new_mem, "add")


# %%
new_message_embeddings[new_mem] = messages_embeddings
existing_memories = memory.vector_store.search(
    query=new_mem,
    vectors=messages_embeddings,
    limit=5,
    filters=search_filters,
)


for mem in existing_memories:
    retrieved_old_memory.append({"id": mem.id, "text": mem.payload.get("data", "")})
unique_data = {}
for item in retrieved_old_memory:
    unique_data[item["id"]] = item
retrieved_old_memory = list(unique_data.values())


# %%


logger.info(f"Total existing memories: {len(retrieved_old_memory)}")

# mapping UUIDs with integers for handling UUID hallucinations
temp_uuid_mapping = {}
for idx, item in enumerate(retrieved_old_memory):
    temp_uuid_mapping[str(idx)] = item["id"]
    retrieved_old_memory[idx]["id"] = str(idx)

if new_retrieved_facts:
    function_calling_prompt = get_update_memory_messages(
        retrieved_old_memory, new_retrieved_facts, memory.config.custom_update_memory_prompt
    )

    try:
        response: str = memory.llm.generate_response(
            messages=[{"role": "user", "content": function_calling_prompt}],
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.error(f"Error in new memory actions response: {e}")
        response = ""
    # response = """{
    #     "memory": [
    #         {
    #             "id": "0",
    #             "text": "Loves sci-fi movies",
    #             "event": "ADD"
    #         },
    #         {
    #             "id": "1", 
    #             "text": "Not a big fan of thriller movies",
    #             "event": "ADD"
    #         }
    #     ]
    # }
    # """
    try:
        if not response or not response.strip():
            logger.warning("Empty response from LLM, no memories to extract")
            new_memories_with_actions = {}
        else:
            response = remove_code_blocks(response)
            new_memories_with_actions = json.loads(response)
    except Exception as e:
        logger.error(f"Invalid JSON response: {e}")
        new_memories_with_actions = {}
else:
    new_memories_with_actions = {}

returned_memories = []
try:
    for resp in new_memories_with_actions.get("memory", []):
        logger.info(resp)
        try:
            action_text = resp.get("text")
            if not action_text:
                logger.info("Skipping memory entry because of empty `text` field.")
                continue

            event_type = resp.get("event")
            if event_type == "ADD":
                memory_id = memory._create_memory(
                    data=action_text,
                    existing_embeddings=new_message_embeddings,
                    metadata=deepcopy(metadata),
                )
                returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type})
            elif event_type == "UPDATE":
                memory._update_memory(
                    memory_id=temp_uuid_mapping[resp.get("id")],
                    data=action_text,
                    existing_embeddings=new_message_embeddings,
                    metadata=deepcopy(metadata),
                )
                returned_memories.append(
                    {
                        "id": temp_uuid_mapping[resp.get("id")],
                        "memory": action_text,
                        "event": event_type,
                        "previous_memory": resp.get("old_memory"),
                    }
                )
            elif event_type == "DELETE":
                memory._delete_memory(memory_id=temp_uuid_mapping[resp.get("id")])
                returned_memories.append(
                    {
                        "id": temp_uuid_mapping[resp.get("id")],
                        "memory": action_text,
                        "event": event_type,
                    }
                )
            elif event_type == "NONE":
                # Even if content doesn't need updating, update session IDs if provided
                memory_id = temp_uuid_mapping.get(resp.get("id"))
                if memory_id and (metadata.get("agent_id") or metadata.get("run_id")):
                    # Update only the session identifiers, keep content the same
                    existing_memory = memory.vector_store.get(vector_id=memory_id)
                    updated_metadata = deepcopy(existing_memory.payload)
                    if metadata.get("agent_id"):
                        updated_metadata["agent_id"] = metadata["agent_id"]
                    if metadata.get("run_id"):
                        updated_metadata["run_id"] = metadata["run_id"]
                    updated_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

                    memory.vector_store.update(
                        vector_id=memory_id,
                        vector=None,  # Keep same embeddings
                        payload=updated_metadata,
                    )
                    logger.info(f"Updated session IDs for memory {memory_id}")
                else:
                    logger.info("NOOP for Memory.")
        except Exception as e:
            logger.error(f"Error processing memory action: {resp}, Error: {e}")
except Exception as e:
    logger.error(f"Error iterating new_memories_with_actions: {e}")

keys, encoded_ids = process_telemetry_filters(filters)
capture_event(
    "mem0.add",
    memory,
    {"version": memory.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"},
)


# %% [markdown]
# `memory.get_all(user_id="alice")`

# %%
limit = 100
_, effective_filters = _build_filters_and_metadata(
    user_id="alice", agent_id=None, run_id=None, input_filters=None
)
if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
    raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")

keys, encoded_ids = process_telemetry_filters(effective_filters)
capture_event(
    "mem0.get_all", memory, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"}
)

with concurrent.futures.ThreadPoolExecutor() as executor:
    future_memories = executor.submit(memory._get_all_from_vector_store, effective_filters, limit)
    future_graph_entities = (
        executor.submit(memory.graph.get_all, effective_filters, limit) if memory.enable_graph else None
    )

    concurrent.futures.wait(
        [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
    )

    all_memories_result = future_memories.result()
    graph_entities_result = future_graph_entities.result() if future_graph_entities else None





