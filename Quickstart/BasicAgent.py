from langchain.agents import create_agent
#from langchain.tools import tool

#@tool
def get_weather(city:str)->str:
    """Get weather for a given city."""
    return f"Its always sunny in {city}"

agent = create_agent(
    model='openai:gpt-5.4',
    tools=[get_weather],
    system_prompt='You are a helpful assistant'
)

result = agent.invoke(
    {'messages':[{'role':'user','content':'What is the weather in San Francisco?'}]}
)

print(result['messages'][-1].content_blocks)