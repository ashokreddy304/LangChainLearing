from langchain_openai import OpenAI
from langchain.messages import SystemMessage,HumanMessage
from dotenv import load_dotenv
load_dotenv()

model = OpenAI(temperature=0.7,top_p=0.9)

syt_message = SystemMessage(
"""
Your as senior ML Engineer. please explain the questions in detailes if user ask other ML you say polite message.
"""
)

humn_message = input("Enter your ML Related Questions only: ")

messages = [syt_message,humn_message]

result = model.invoke(messages)
print(result)
