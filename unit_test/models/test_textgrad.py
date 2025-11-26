
from textgrad import Variable
from textgrad.optimizer import TextualGradientDescent
from textgrad.loss import TextLoss

import sys, os
root_dir = os.getcwd()
sys.path.append(root_dir)
from langagent.llm.openai_model import AnyOpenAILLM
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from textgrad.engine.base import EngineLM
import textgrad as tg


user_input = Variable("A sntence with a typo", role_description="The input sentence", requires_grad=True)
class ChatOpenAI(EngineLM):

    def __init__(self, model_string="chatgpt-4k"):
        super().__init__()
        self.client = AnyOpenAILLM( model_string)
        self.model_string = model_string

    def generate(self, usr_prompt, system_prompt=None, **kwargs):
        system_prompt = system_prompt if system_prompt else self.system_prompt
        response = self.client(usr_prompt, system_prompt, **kwargs)        
        return response
    
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
    


engine = ChatOpenAI("chatgpt-4k") # "chatgpt-16k", "gpt4-short", "gpt-3.5-turbo-0125"

# model = tg.BlackboxLLM(engine, system_prompt)
# # print(engine.generate("Generate one good word?"))

# system_prompt = Variable("Evaluate the correctness of this sentence", role_description="The system prompt")
# loss = TextLoss(system_prompt, engine=engine)
# l = loss(user_input) # Variable(value=The sentence is incorrect due to the typo., role=response from the language model, grads=)
# print('Loss before backward:', l)
# l.backward(engine)
# print('Loss after backward:', l)
# print('LLM Response after backward:', )

# optimizer = TextualGradientDescent(parameters=[user_input], engine=engine)
# optimizer.step()




# Load the data and the evaluation function (train_set, val_set, test_set, eval_fn)
from textgrad.tasks import load_task
from textgrad.tasks.gsm8k import GSM8K_DSPy
from textgrad.tasks.big_bench_hard import string_based_equality_fn
from textgrad.autograd.string_based_ops import StringBasedFunction
import platformdirs
    
class GSM8K_DSPy():
    """ Modifying https://github.com/zou-group/textgrad/blob/main/textgrad/tasks/gsm8k.py to solve the following error (version 0.1.4) & Inheritance (GSM8K_DSPy(GSM8K)) is removed:
        TypeError: Can't instantiate abstract class GSM8K_DSPy without an implementation for abstract method 'get_default_task_instruction'
    """
    def __init__(self, root:str=None, split: str="train"):
        """DSPy splits for the GSM8K dataset."""
        import tqdm
        import random
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        dataset = load_dataset("gsm8k", 'main', cache_dir=root)
        hf_official_train = dataset['train']
        hf_official_test = dataset['test']
        official_train = []
        official_test = []
        for example in tqdm.tqdm(hf_official_train):
            question = example['question']
            answer = example['answer'].strip().split()
            assert answer[-2] == '####'
            
            gold_reasoning = ' '.join(answer[:-2])
            answer = str(int(answer[-1].replace(',', '')))
            official_train.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))

        for example in tqdm.tqdm(hf_official_test):
            question = example['question']
            answer = example['answer'].strip().split()
            assert answer[-2] == '####'
            
            gold_reasoning = ' '.join(answer[:-2])
            answer = str(int(answer[-1].replace(',', '')))
            official_test.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))

        rng = random.Random(0)
        rng.shuffle(official_train)
        rng = random.Random(0)
        rng.shuffle(official_test)
        trainset = official_train[:200]
        devset = official_train[200:500]
        testset = official_test[:]
        if split == "train":
            self.data = trainset
        elif split == "val":
            self.data = devset
        elif split == "test":
            self.data = testset

    def __getitem__(self, index):
        row = self.data[index]
        question = row["question"]
        answer = row["answer"]
        question_prompt = f"Question: {question}"
        return question_prompt, answer

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return "You will answer a mathemetical reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
    

train_set = GSM8K_DSPy(split="train")
val_set = GSM8K_DSPy(split="val")
test_set = GSM8K_DSPy(split="test")
fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
eval_fn = StringBasedFunction(string_based_equality_fn, function_purpose=fn_purpose)

# model
STARTING_SYSTEM_PROMPT = train_set.get_task_description()
system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                            requires_grad=True,
                            role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")
model = tg.BlackboxLLM(engine, system_prompt)

# train
train_loader = tg.tasks.DataLoader(train_set, batch_size=3, shuffle=True)
for batch_x, batch_y in train_loader:
    losses = []
    for (x, y) in zip(batch_x, batch_y):
        x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
        y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
        response = model(x)
        try:
            eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        except:
            eval_output_variable = eval_fn([x, y, response])
        losses.append(eval_output_variable)
    total_loss = tg.sum(losses)
    total_loss.backward(engine)
    break