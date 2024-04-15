# from langchain.llms import HuggingFacePipeline

# # model_id = "tiiuae/falcon-7b-instruct"
# model_id = "lmsys/fastchat-t5-3b-v1.0"


# hf = HuggingFacePipeline.from_model_id(
#     model_id=model_id,
#     task="text-generation",
#     pipeline_kwargs={"max_new_tokens": 10},
# )

# hf("Hello!")

# # Use a pipeline as a high-level helper
# # from transformers import pipeline

# # pipe = pipeline("text2text-generation", model="lmsys/fastchat-t5-3b-v1.0")

from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

model_id = "lmsys/fastchat-t5-3b-v1.0"
llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text2text-generation",
    model_kwargs={
        "temperature": 0.01,
        "max_length": 1000,
        'do_sample': True
    },
)

# template = """
# You are a friendly chatbot assistant that responds conversationally to users' questions.
# Keep the answers short, unless specifically asked by the user to elaborate on something.
# You always answer in spanish.

# Question: {question}

# Answer:"""

template = """
Eres un Mexicano experto en hablar profesional.

Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

result = llm_chain("Dame un consejo cual sea")
print(result['text'])
