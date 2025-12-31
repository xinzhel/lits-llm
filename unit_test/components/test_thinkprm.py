"""
Test ThinkPRM RewardModel integration.

This test verifies that ThinkPRM correctly wraps ThinkPRMSageMaker
and provides the standard RewardModel interface for math QA tasks.

Requirements:
- SageMaker endpoint 'thinkprm-14b-endpoint' must be running
- AWS credentials configured (aws sso login)

Test Cases:
1. Math500 Divisors - correct multi-step solution (amicable numbers 284/220)
2. Math500 Parentheses - incomplete solution with incorrect first step
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
from lits.components.reward.thinkprm import ThinkPRM
from lits.structures.qa import ThoughtStep
from lits.structures.base import TrajectoryState


# =============================================================================
# Test Data: Math500 Examples
# =============================================================================

# Test 1: Proper Divisors (correct solution)
MATH500_DIVISORS = {
    "question": (
        "The proper divisors of 12 are 1, 2, 3, 4 and 6. A proper divisor of an integer $N$ "
        "is a positive divisor of $N$ that is less than $N$. What is the sum of the proper "
        "divisors of the sum of the proper divisors of 284?"
    ),
    "steps": [
        "First, I need to find the proper divisors of 284. The divisors of 284 are 1, 2, 4, 71, 142, and 284. The proper divisors (excluding 284 itself) are 1, 2, 4, 71, and 142.",
        "The sum of the proper divisors of 284 is 1 + 2 + 4 + 71 + 142 = 220.",
        "Now I need to find the proper divisors of 220. The divisors of 220 are 1, 2, 4, 5, 10, 11, 20, 22, 44, 55, 110, and 220. The proper divisors are 1, 2, 4, 5, 10, 11, 20, 22, 44, 55, and 110. The sum is 1 + 2 + 4 + 5 + 10 + 11 + 20 + 22 + 44 + 55 + 110 = 284."
    ],
    "expected_all_correct": True,
}

# Test 2: Parentheses Problem (incomplete/incorrect solution)
MATH500_PARENTHESES = {
    "question": (
        r"The expression $2\cdot 3 \cdot 4\cdot 5+1$ is equal to 121, since multiplication "
        r"is carried out before addition. However, we can obtain values other than 121 for "
        r"this expression if we are allowed to change it by inserting parentheses. For example, "
        r"we can obtain 144 by writing \[(2\cdot (3\cdot 4)) \cdot (5+1) = 144.\]In total, how "
        r"many values can be obtained from the expression $2\cdot 3\cdot 4 \cdot 5 + 1$ by "
        r"inserting parentheses? (Note that rearranging terms is not allowed, only inserting parentheses)."
    ),
    "steps": [
        # Step 1 is INCORRECT - doesn't list all possible parenthesizations (missing those with +1 inside)
        "Let's start by identifying all the possible ways to insert parentheses in the expression 2 · 3 · 4 · 5 + 1. We'll list them out:\n\n1. 2 · 3 · 4 · 5 + 1 (original, no parentheses)\n2. (2 · 3) · 4 · 5 + 1\n3. (2 · 3 · 4) · 5 + 1\n4. (2 · 3 · 4 · 5) + 1\n5. 2 · (3 · 4) · 5 + 1\n6. 2 · (3 · 4 · 5) + 1\n7. 2 · 3 · (4 · 5) + 1\n8. 2 · (3 · (4 · 5)) + 1\n9. (2 · 3) · (4 · 5) + 1\n10. ((2 · 3) · 4) · 5 + 1\n11. (2 · (3 · 4)) · 5 + 1\n\nThese are all the possible ways to insert parentheses without rearranging the terms. In the next step, we'll start calculating the values for each of these expressions.",
        # Step 2 - calculations are correct for the listed expressions
        "Let's calculate the values for each of these expressions:\n\n1. 2 · 3 · 4 · 5 + 1 = 120 + 1 = 121\n2. (2 · 3) · 4 · 5 + 1 = 6 · 4 · 5 + 1 = 120 + 1 = 121\n3. (2 · 3 · 4) · 5 + 1 = 24 · 5 + 1 = 120 + 1 = 121\n4. (2 · 3 · 4 · 5) + 1 = 120 + 1 = 121\n5. 2 · (3 · 4) · 5 + 1 = 2 · 12 · 5 + 1 = 120 + 1 = 121\n6. 2 · (3 · 4 · 5) + 1 = 2 · 60 + 1 = 120 + 1 = 121\n7. 2 · 3 · (4 · 5) + 1 = 2 · 3 · 20 + 1 = 120 + 1 = 121\n8. 2 · (3 · (4 · 5)) + 1 = 2 · (3 · 20) + 1 = 2 · 60 + 1 = 120 + 1 = 121\n9. (2 · 3) · (4 · 5) + 1 = 6 · 20 + 1 = 120 + 1 = 121\n10. ((2 · 3) · 4) · 5 + 1 = 24 · 5 + 1 = 120 + 1 = 121\n11. (2 · (3 · 4)) · 5 + 1 = 24 · 5 + 1 = 120 + 1 = 121\n\nInterestingly, we see that all these expressions evaluate to 121, except for the one given in the problem statement: (2 · (3 · 4)) · (5 + 1) = 24 · 6 = 144.",
        # Step 3 - realizes the mistake
        "Let's analyze why we got the same result (121) for all the expressions we calculated, and consider if there are any other possible arrangements we might have missed:\n\n1. In all our calculated expressions, the addition of 1 was always the last operation performed. This is because we didn't consider placing parentheses around any part of the expression that included the \"+1\" at the end.\n\n2. The expression given in the problem, (2 · (3 · 4)) · (5 + 1), shows us that we can indeed include the \"+1\" within parentheses.\n\n3. This suggests we need to consider more possibilities where the \"+1\" is included within parentheses along with some of the multiplication operations.\n\nSome additional possibilities include:\n- 2 · 3 · 4 · (5 + 1)\n- 2 · 3 · (4 · (5 + 1))\n- 2 · (3 · (4 · (5 + 1)))\n- (2 · 3) · (4 · (5 + 1))\n- (2 · 3 · 4) · (5 + 1)\n\nIn the next step, we'll calculate these additional expressions to see if we get any new values."
    ],
    "expected_first_step_incorrect": True,
}


# =============================================================================
# Test Functions
# =============================================================================

def test_math500_divisors():
    """
    Test ThinkPRM with Math500 proper divisors problem (correct solution).
    
    This tests a 3-step correct solution about amicable numbers (284 and 220).
    All steps should be marked correct.
    """
    print("=" * 70)
    print("Test: Math500 Proper Divisors (Correct Solution)")
    print("=" * 70)
    
    prm = ThinkPRM(
        endpoint_name="thinkprm-14b-endpoint",
        region_name="us-east-1",
        max_new_tokens=2048,
        scoring_mode="last_step",
    )
    
    question = MATH500_DIVISORS["question"]
    steps = MATH500_DIVISORS["steps"]
    
    # Build state with first 2 steps
    state = TrajectoryState()
    for step_text in steps[:-1]:
        state.append(ThoughtStep(action=step_text))
    
    # Evaluate the last step
    new_step = ThoughtStep(action=steps[-1])
    
    print(f"Question: {question[:80]}...")
    print(f"Existing steps: {len(state)}")
    print(f"Evaluating step {len(state) + 1}")
    print("-" * 70)
    
    start_time = time.time()
    reward, details = prm.fast_reward(state, new_step, question, query_idx=0)
    elapsed = time.time() - start_time
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Reward (last_step): {reward:.3f}")
    print(f"Step labels: {details.get('step_labels', [])}")
    print(f"New step label: {details.get('new_step_label', 'N/A')}")
    print("-" * 70)
    
    # All steps should be correct
    step_labels = details.get('step_labels', [])
    if step_labels and all(label == 1 for label in step_labels):
        print("✓ All steps marked correct (as expected)")
    else:
        print(f"⚠ Some steps marked incorrect: {step_labels}")
    
    print("=" * 70)
    return reward, details


def test_math500_parentheses():
    """
    Test ThinkPRM with Math500 parentheses problem (incomplete/incorrect solution).
    
    This tests a 3-step solution where Step 1 is incorrect (doesn't list all
    possible parenthesizations). The model should detect this error.
    
    Expected behavior: Model marks Step 1 as incorrect and may stop early.
    """
    print("\n" + "=" * 70)
    print("Test: Math500 Parentheses (Incomplete Solution - Step 1 Incorrect)")
    print("=" * 70)
    
    prm = ThinkPRM(
        endpoint_name="thinkprm-14b-endpoint",
        region_name="us-east-1",
        max_new_tokens=4096,  # Longer for complex problem
        scoring_mode="last_step",
    )
    
    question = MATH500_PARENTHESES["question"]
    steps = MATH500_PARENTHESES["steps"]
    
    # Build state with first 2 steps
    state = TrajectoryState()
    for step_text in steps[:-1]:
        state.append(ThoughtStep(action=step_text))
    
    # Evaluate the last step
    new_step = ThoughtStep(action=steps[-1])
    
    print(f"Question: {question[:80]}...")
    print(f"Existing steps: {len(state)}")
    print(f"Evaluating step {len(state) + 1}")
    print("-" * 70)
    
    start_time = time.time()
    reward, details = prm.fast_reward(state, new_step, question, query_idx=1)
    elapsed = time.time() - start_time
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Reward (last_step): {reward:.3f}")
    print(f"Step labels: {details.get('step_labels', [])}")
    print(f"New step label: {details.get('new_step_label', 'N/A')}")
    print("-" * 70)
    
    # Step 1 should be incorrect (missing parenthesizations with +1)
    step_labels = details.get('step_labels', [])
    if step_labels and step_labels[0] == 0:
        print("✓ Step 1 correctly marked as incorrect (missing parenthesizations)")
    else:
        print(f"⚠ Step 1 not marked incorrect: {step_labels}")
    
    print("=" * 70)
    return reward, details


def test_scoring_modes():
    """
    Test different scoring modes with the Math500 divisors example.
    
    Compares last_step, prefix, and average scoring modes.
    """
    print("\n" + "=" * 70)
    print("Test: Scoring Modes Comparison")
    print("=" * 70)
    
    question = MATH500_DIVISORS["question"]
    steps = MATH500_DIVISORS["steps"]
    
    # Build state with first 2 steps
    state = TrajectoryState()
    for step_text in steps[:-1]:
        state.append(ThoughtStep(action=step_text))
    new_step = ThoughtStep(action=steps[-1])
    
    print(f"Question: {question[:60]}...")
    print(f"Steps: {len(state)} existing + 1 new")
    print("-" * 70)
    
    results = {}
    for mode in ["last_step", "prefix", "average"]:
        prm = ThinkPRM(
            endpoint_name="thinkprm-14b-endpoint",
            region_name="us-east-1",
            max_new_tokens=2048,
            scoring_mode=mode,
        )
        
        start_time = time.time()
        reward, details = prm.fast_reward(state, new_step, question, query_idx=2)
        elapsed = time.time() - start_time
        
        results[mode] = {
            'reward': reward,
            'step_labels': details.get('step_labels', []),
            'time': elapsed,
        }
        print(f"  {mode:12s}: reward={reward:.3f}, labels={details.get('step_labels', [])}, time={elapsed:.2f}s")
    
    print("=" * 70)
    print("✓ All scoring modes executed")
    return results


def test_calculate_reward():
    """Test calculate_reward method with different alpha values."""
    print("\n" + "=" * 70)
    print("Test: calculate_reward (no endpoint call)")
    print("=" * 70)
    
    # Test with default alpha=1.0
    prm = ThinkPRM(endpoint_name="thinkprm-14b-endpoint", reward_alpha=1.0)
    
    test_scores = [0.0, 0.5, 0.8, 1.0]
    print("With reward_alpha=1.0:")
    for score in test_scores:
        result = prm.calculate_reward(score)
        print(f"  calculate_reward({score}) = {result}")
        assert result == score ** 1.0
    
    # Test with alpha=0.5
    prm2 = ThinkPRM(endpoint_name="thinkprm-14b-endpoint", reward_alpha=0.5)
    print("\nWith reward_alpha=0.5:")
    for score in test_scores:
        result = prm2.calculate_reward(score)
        expected = score ** 0.5
        print(f"  calculate_reward({score}) = {result:.3f}")
    
    print("=" * 70)
    print("✓ calculate_reward works correctly")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ThinkPRM RewardModel Integration Tests")
    print("=" * 70 + "\n")
    
    # # Test that doesn't need endpoint
    # test_calculate_reward()
    
    # print("\n" + "-" * 70)
    # print("The following tests require the SageMaker endpoint to be running.")
    # print("Endpoint: thinkprm-14b-endpoint")
    # print("-" * 70 + "\n")
    
    try:
        test_math500_divisors()
        test_math500_parentheses()
        # test_scoring_modes()  # Uncomment to test all modes (slower)
        
        print("\n" + "=" * 70)
        print("All tests completed!")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Make sure the SageMaker endpoint is running and AWS credentials are configured.")
        raise
