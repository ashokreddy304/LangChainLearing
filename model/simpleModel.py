from dotenv import load_dotenv
load_dotenv()
import os
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model='gpt-5.4'
)

response = model.invoke('what is the capital of india?')

print(response.content)

#Partially configurable model with no default
configurable_model = init_chat_model(temperature=0)

result = configurable_model.invoke(
    'what is the capital of india?',
    config={'configurable':{'model':'gpt-5.5'}}
)
print(result.content)