from transformers import AutoTokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct" #
tokenizer = AutoTokenizer.from_pretrained(model_name)

example = "3+5*6="
usr_prompt = """Given a science problem, your task is to answer the question step-by-step in a clear and specific manner.
The format of the solution is limited to: "Solution: ...\nSummary: The final answer is $...$"
Please complete the answer step-by-step, and finally outline the final answer.
Problem: """ + example + "\nSolution:"



def tokenize_one_usr_prompt(usr_prompt):
    messages = [
        # {"role": "system", "content": self.sys_prompt},
        {"role": "user", "content": usr_prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )

    return text

# ['<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nGiven a science problem, your task is to answer the question step-by-step in a clear and specific manner.\nThe format of the solution is limited to: "Solution: ...\nSummary: The final answer is $...$"\nPlease complete the answer step-by-step, and finally outline the final answer.\nProblem: 3+5*6=\nSolution:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n']