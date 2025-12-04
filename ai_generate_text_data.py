import os

from dotenv import load_dotenv
from langchain_experimental.synthetic_data import create_data_generation_chain
from langchain_openai import ChatOpenAI

load_dotenv()

# 创建模型
model = ChatOpenAI(model=os.getenv("MODEL_NAME"),
                   api_key=os.getenv("OPENAI_API_KEY"),
                   base_url=os.getenv("BASE_URL"),
                   temperature=0.8)

# 创建链
chain = create_data_generation_chain(model)  # 创建生成数据的链

# 生成数据
result = chain.invoke({ # 给定一些关键词，随机生成一句话
    "fields": ['蓝色', '黄色'],  # fields是固定的
    "preferences": {} # 字典参数，用于传递用户的偏好或特定设置，此时没有配置为空
})

print(result)

result = chain.invoke({ # 给定一些关键词，随机生成一句话
    "fields": ['蓝色', '黄色'],  # fields是固定的
    "preferences": {"stype":"诗歌风格"} # 添加风格设置
})

print(result)