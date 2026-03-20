"""Test script for lits.embedding subpackage.

Tests:
1. SentenceTransformerEmbedder — shape, dtype, L2-norm, cosine sim ordering
2. BedrockEmbedder (Cohere) — shape, dtype, L2-norm, cosine sim
3. BedrockEmbedder (Titan) — shape, dtype, L2-norm
4. get_embedder() dispatch — correct backend class returned

NOTE: Bedrock tests (2, 3) require valid AWS credentials.
      Test has NOT been run yet.
"""

import numpy as np


# ── Shared helpers ────────────────────────────────────────────────────

TEXTS = [
    "The prime factorization of 196 is 2 squared times 7 squared.",
    "196 equals 2^2 * 7^2, which is its prime decomposition.",
    "The weather in Melbourne is often unpredictable.",
]


def print_norms(vecs: np.ndarray, label: str) -> None:
    norms = np.linalg.norm(vecs, axis=1)
    print(f"\n[{label}] L2 norms (should all be ≈1.0):")
    for i, n in enumerate(norms):
        print(f"  row {i}: {n:.6f}")


def print_cosine_matrix(vecs: np.ndarray, label: str) -> None:
    # Since vectors are L2-normalised, dot product == cosine similarity
    sim = vecs @ vecs.T
    print(f"\n[{label}] Cosine similarity matrix:")
    print(f"  Texts:")
    for i, t in enumerate(TEXTS):
        print(f"    [{i}] {t[:60]}...")
    print(f"  Matrix:")
    for i in range(sim.shape[0]):
        row = "  ".join(f"{sim[i, j]:.4f}" for j in range(sim.shape[1]))
        print(f"    [{i}] {row}")
    print(f"  sim(0,1) = {sim[0,1]:.4f}  (similar facts — should be high)")
    print(f"  sim(0,2) = {sim[0,2]:.4f}  (unrelated — should be lower)")


def pause(msg: str = "Press Enter to continue...") -> None:
    input(f"\n>>> {msg}")


# ── Test 1: SentenceTransformerEmbedder ───────────────────────────────

def test_sentence_transformer():
    print("\n" + "=" * 70)
    print("TEST 1: SentenceTransformerEmbedder")
    print("=" * 70)

    from lits.embedding.sentence_transformer import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    print(f"Model: multi-qa-mpnet-base-cos-v1")
    print(f"embedding_dim: {embedder.embedding_dim}")

    vecs = embedder.embed(TEXTS)
    print(f"shape: {vecs.shape}  (expect (3, {embedder.embedding_dim}))")
    print(f"dtype: {vecs.dtype}  (expect float32)")

    print_norms(vecs, "SentenceTransformer")
    print_cosine_matrix(vecs, "SentenceTransformer")
    pause("Inspect norms and cosine matrix above. Press Enter to continue...")


# ── Test 2: BedrockEmbedder (Cohere) ──────────────────────────────────

def test_bedrock_cohere():
    print("\n" + "=" * 70)
    print("TEST 2: BedrockEmbedder (Cohere)")
    print("=" * 70)

    from lits.embedding.bedrock import BedrockEmbedder

    embedder = BedrockEmbedder(model_id="cohere.embed-english-v3")
    print(f"Model: cohere.embed-english-v3")
    print(f"embedding_dim: {embedder.embedding_dim}")

    vecs = embedder.embed(TEXTS)
    print(f"shape: {vecs.shape}  (expect (3, {embedder.embedding_dim}))")
    print(f"dtype: {vecs.dtype}  (expect float32)")

    print_norms(vecs, "Bedrock-Cohere")
    print_cosine_matrix(vecs, "Bedrock-Cohere")
    pause("Inspect norms and cosine matrix above. Press Enter to continue...")


# ── Test 3: BedrockEmbedder (Titan) ───────────────────────────────────

def test_bedrock_titan():
    print("\n" + "=" * 70)
    print("TEST 3: BedrockEmbedder (Titan)")
    print("=" * 70)

    from lits.embedding.bedrock import BedrockEmbedder

    embedder = BedrockEmbedder(
        model_id="amazon.titan-embed-text-v2:0", dimensions=1024
    )
    print(f"Model: amazon.titan-embed-text-v2:0")
    print(f"embedding_dim: {embedder.embedding_dim}")

    vecs = embedder.embed(TEXTS)
    print(f"shape: {vecs.shape}  (expect (3, 1024))")
    print(f"dtype: {vecs.dtype}  (expect float32)")

    print_norms(vecs, "Bedrock-Titan")
    print_cosine_matrix(vecs, "Bedrock-Titan")
    pause("Inspect norms and cosine matrix above. Press Enter to continue...")


# ── Test 4: get_embedder() dispatch ───────────────────────────────────

def test_get_embedder_dispatch():
    print("\n" + "=" * 70)
    print("TEST 4: get_embedder() dispatch")
    print("=" * 70)

    from lits.embedding import get_embedder
    from lits.embedding.sentence_transformer import SentenceTransformerEmbedder
    from lits.embedding.bedrock import BedrockEmbedder

    # ST — reuse the already-loaded model if test_sentence_transformer ran
    try:
        st = get_embedder("multi-qa-mpnet-base-cos-v1")
        print(f"get_embedder('multi-qa-mpnet-base-cos-v1') → {type(st).__name__}")
        print(f"  Is SentenceTransformerEmbedder? {isinstance(st, SentenceTransformerEmbedder)}")
    except Exception as e:
        print(f"  [SKIP] SentenceTransformer dispatch: {e}")

    # Bedrock — will fail without AWS creds, but we can still check the type
    try:
        br = get_embedder("bedrock-embed/cohere.embed-english-v3")
        print(f"get_embedder('bedrock-embed/cohere.embed-english-v3') → {type(br).__name__}")
        print(f"  Is BedrockEmbedder? {isinstance(br, BedrockEmbedder)}")
    except Exception as e:
        print(f"  [SKIP] Bedrock dispatch: {e}")

    pause("Inspect dispatch results above. Press Enter to finish.")


# ── Test 5: LocalMemoryBackend with Bedrock embedder ──────────────────

def test_local_backend_with_bedrock_embedder():
    print("\n" + "=" * 70)
    print("TEST 5: LocalMemoryBackend with Bedrock embedder (dedup)")
    print("=" * 70)

    from lits.embedding import get_embedder
    from lits.lm import get_lm
    from lits.memory.backends import LocalMemoryBackend
    from lits.memory.types import TrajectoryKey

    MODEL_NAME = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
    llm = get_lm(MODEL_NAME)
    embedder = get_embedder("bedrock-embed/cohere.embed-english-v3")

    backend = LocalMemoryBackend(llm=llm, embedder=embedder)
    print(f"Backend created with Bedrock Cohere embedder (dim={embedder.embedding_dim})")

    # Two semantically similar facts from different trajectories
    traj_a = TrajectoryKey(search_id="test_search", indices=(0,))
    traj_b = TrajectoryKey(search_id="test_search", indices=(1,))

    msgs_a = [{"role": "assistant", "content": "196 = 2^2 * 7^2"}]
    msgs_b = [{"role": "assistant", "content": "The prime factorization of 196 is 2 squared times 7 squared."}]

    units_a = backend.add_messages(traj_a, msgs_a, metadata={}, infer=False, query_idx=0)
    print(f"\nTrajectory A added {len(units_a)} unit(s):")
    for u in units_a:
        print(f"  text={u.text!r}  hash={u.content_hash[:12]}...")

    units_b = backend.add_messages(traj_b, msgs_b, metadata={}, infer=False, query_idx=0)
    print(f"\nTrajectory B added {len(units_b)} unit(s):")
    for u in units_b:
        print(f"  text={u.text!r}  hash={u.content_hash[:12]}...")

    all_units = backend.list_all_units("test_search")
    print(f"\nAll units for search_id='test_search': {len(all_units)}")
    for u in all_units:
        print(f"  origin={u.origin_path}  hash={u.content_hash[:12]}...  text={u.text!r}")

    # Check signature overlap (should match if dedup aliased correctly)
    sigs_a = {u.signature() for u in all_units if u.origin_path == "q/0"}
    sigs_b = {u.signature() for u in all_units if u.origin_path == "q/1"}
    overlap = sigs_a & sigs_b
    print(f"\nSignature overlap: {len(overlap)} (should be 1 if dedup worked)")
    print(f"  A sigs: {sigs_a}")
    print(f"  B sigs: {sigs_b}")

    pause("Inspect dedup results above. Press Enter to finish.")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # # Test 1 runs locally (no AWS needed)
    # test_sentence_transformer()

    # # Tests 2-3 require AWS credentials
    # try:
    #     test_bedrock_cohere()
    # except Exception as e:
    #     print(f"\n[SKIP] test_bedrock_cohere: {e}")

    # try:
    #     test_bedrock_titan()
    # except Exception as e:
    #     print(f"\n[SKIP] test_bedrock_titan: {e}")

    # # Test 4: dispatch (Bedrock init may fail without creds, but ST works)
    # test_get_embedder_dispatch()

    # Test 5: LocalMemoryBackend with Bedrock embedder (requires AWS creds + LLM)
    try:
        test_local_backend_with_bedrock_embedder()
    except Exception as e:
        print(f"\n[SKIP] test_local_backend_with_bedrock_embedder: {e}")

    print("\n" + "=" * 70)
    print("All tests completed.")
    print("=" * 70)
