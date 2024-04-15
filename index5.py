from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch

# model = "tiiuae/falcon-40b-instruct"
# model = "stabilityai/stablelm-tuned-alpha-3b"
model = "lmsys/fastchat-t5-3b-v1.0"


tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)


llm_falcon = HuggingFacePipeline(
    pipeline=pipeline,
    model_kwargs={
        'temperature': 0,
        'max_length': 20,
        # 'do_sample': True,
        'top_k': 10,
        'num_return_sequences': 1,
        'eos_token_id': tokenizer.eos_token_id
    }
)

resp = llm_falcon("Hello")
print(resp)
