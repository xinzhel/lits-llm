
import sys, os
root_dir = os.getcwd()
sys.path.append(root_dir)

from langagent.llm.hf_model import HFModel

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
model_name = "../Qwen/Qwen2-1.5B" 
input_text = "Write me a poem about Machine Learning."
model = HFModel(model_name, device="mps")

output_text = model.generate([input_text])
print(output_text.text)





