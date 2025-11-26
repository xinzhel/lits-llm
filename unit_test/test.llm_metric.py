import sys
sys.path.append('.')
from lits.lm import InferenceLogger
import os

if  __name__ == "__main__":

    # Instantiate logger
    logger = InferenceLogger()

    # 1) Single non-batch call for "actor"
    logger.update_usage(input_tokens=10, output_tokens=20, batch=False, batch_size=1, role="actor")
    print("After 1st actor call, Overall metrics:", logger.get_metrics())
    print("\tActor metrics:", logger.get_metrics(role="actor"))
    print("\tEvaluator metrics:", logger.get_metrics(role="evaluator"))

    # 2) Single non-batch call for "evaluator"
    logger.update_usage(input_tokens=30, output_tokens=40, batch=False, batch_size=1, role="evaluator")
    print("After 1st evaluator call,  Overall metrics:", logger.get_metrics())
    print("\tActor metrics:", logger.get_metrics(role="actor"))
    print("\tEvaluator metrics:", logger.get_metrics(role="evaluator"))

    # 3) Another non-batch call for "actor"
    logger.update_usage(input_tokens=5, output_tokens=5, batch=False, batch_size=1, role="actor")
    print("After 2nd actor call, Overall metrics:", logger.get_metrics())
    print("\tActor metrics:", logger.get_metrics(role="actor"))
    print("\tEvaluator metrics:", logger.get_metrics(role="evaluator"))

    # 4) A batched call for "actor" with batch_size=4
    logger.update_usage(input_tokens=100, output_tokens=150, batch=True, batch_size=4, role="actor")
    print("After batched actor call, Overall metrics:", logger.get_metrics())
    print("\tActor metrics:", logger.get_metrics(role="actor"))
    print("\tEvaluator metrics:", logger.get_metrics(role="evaluator"))
    # Cleanup
    # os.remove(log_path)
