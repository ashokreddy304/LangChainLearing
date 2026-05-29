from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,SystemMessage,AIMessage

from dotenv import load_dotenv
load_dotenv()

model = init_chat_model(model='gpt-5-nano')

System_Message = SystemMessage('You are help full assistance')
Human_Message = HumanMessage("Hello, How are you")

messages = [System_Message,Human_Message]

results = model.invoke(messages)
print(results.content)