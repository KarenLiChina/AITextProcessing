import os

from dotenv import load_dotenv
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# 创建模型
model = ChatOpenAI(model=os.getenv("MODEL_NAME"),
                   api_key=os.getenv("OPENAI_API_KEY"),
                   base_url=os.getenv("BASE_URL"),
                   temperature=0)

loader = WebBaseLoader(web_paths=['https://blog.csdn.net/2301_82275412/article/details/148773003'])
docs = loader.load()  # 得到整篇文字
# stuff第一种写法，没有提示词
chain = load_summarize_chain(model, chain_type='stuff')  # 得到summarize的chain

result = chain.invoke(docs)
print(result['output_text'])  # 'output_text'是固定的输出的key
print('--' * 40)
# stuff第二种的写法，定义提示词
prompt_template = """
针对下面的内容，写一个简介的总结摘要：
"{text}"
简介的总结摘要：
"""
prompt = PromptTemplate.from_template(prompt_template)

stuff_chain = create_stuff_documents_chain(llm=model, prompt=prompt, document_variable_name='text')
result = chain.invoke(docs)
print(result['output_text'])
