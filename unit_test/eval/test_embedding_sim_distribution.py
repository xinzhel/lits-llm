"""Test cosine similarity distribution for semantically equivalent vs non-equivalent pairs.

Empirical results with multi-qa-mpnet-base-cos-v1:
  Equivalent pairs:   0.68 – 0.85
  Non-equivalent:     0.22 – 0.81 (0.81 = "answer is 42" vs "answer is 7")

This justifies threshold=0.65 for Stage 1 embedding pre-filter.

Source: llm_call_logger.py::cluster_by_embedding
"""
from lits.embedding import get_embedder


def test_sim_distribution():
    embedder = get_embedder("multi-qa-mpnet-base-cos-v1")

    pairs = [
        # Equivalent
        ("Let x = 5", "Set x to 5"),
        ("Let x = 5", "Define x as 5"),
        ("Let x = 5", "We assign the value 5 to x"),
        ("Compute y = x^2", "Calculate the square of x to get y"),
        ("Add 3 to both sides", "Adding 3 to each side of the equation"),
        ("Substitute x = 2 into the equation", "Plugging in x = 2"),
        ("The answer is 42", "Therefore the result is 42"),
        ("Simplify the expression", "We can simplify this"),
        # Non-equivalent
        ("Let x = 5", "Compute y = x^2"),
        ("Add 3 to both sides", "Multiply both sides by 2"),
        ("The answer is 42", "The answer is 7"),
        ("Substitute x = 2", "Factor the polynomial"),
    ]

    all_texts = []
    for a, b in pairs:
        all_texts.extend([a, b])

    vecs = embedder.embed(all_texts)
    sim = vecs @ vecs.T

    print("Equivalent pairs:")
    for i in range(8):
        s = sim[i * 2, i * 2 + 1]
        print(f"  {s:.3f}  {pairs[i][0]!r}  vs  {pairs[i][1]!r}")

    print("\nNon-equivalent pairs:")
    for i in range(8, 12):
        s = sim[i * 2, i * 2 + 1]
        print(f"  {s:.3f}  {pairs[i][0]!r}  vs  {pairs[i][1]!r}")

    breakpoint()  # inspect: sim matrix, equivalent range, non-equivalent range


if __name__ == "__main__":
    test_sim_distribution()
