import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "state-spaces/mamba-130m-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

model.eval()

prompt = "Corn disease prediction depends on temperature, humidity, rainfall and leaf wetness."

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7
    )

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)