task_prompt_spec_math_qa = """Your task is to give the correct next step, given a science problem and an existing partial solution (not a complete answer). 
Assuming the input is n-steps, then the format of the input is:
"Problem: ...
Existing Steps:
Step 1: ...
Step 2: ...
...
Step n: ..."
where ... denotes omitted input information. 

Please follow the restricted output format:
* If no existing steps are given, generate Step 1.
* Otherwise, following the given step(s), output ONLY ONE step within 1000 tokens. SHOULD NOT output multiple steps.
* DO NOT repeat Problem or any Existing Steps.
* Your output should be a complete reasoning step that includes calculations, reasoning, choosing answers, etc. 
* When the final answer is/has been reached, begin the step with EXACTLY the phrase:\"Now we can answer the question: The answer is \", followed by EXACTLY one number. Do not include any other words, punctuation, or explanation after the number."""