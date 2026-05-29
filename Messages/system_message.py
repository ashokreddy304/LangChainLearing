from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
load_dotenv()

model = init_chat_model('gpt-5-nano',temperature=0.7)

syst_message = SystemMessage(""""
You are a senior Python developer with expertise in web frameworks.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.""")

hunamn_message = HumanMessage("How do I create a REST API?")

messages = [syst_message,hunamn_message]

result = model.invoke(messages)
print(result.content)