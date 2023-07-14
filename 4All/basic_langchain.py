from langchain import PromptTemplate, LLMChain
from gpt4all import GPT4All

model_name = "orca-mini-3b.ggmlv3.q4_0.bin"
template = """Question: {question}

Answer: Let's think step by step.
"""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = GPT4All(model_name)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = input("Enter your question: ")

llm_chain.run(question)
