from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


def chat(state: AgentState)-> AgentState:
    """Solve the requests from the user input"""
    response = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node('chat', chat)
graph.add_edge(START, 'chat')
graph.add_edge('chat', END)

agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")

while user_input != 'exit':
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    print(result["messages"])
    conversation_history = result['messages']

    user_input = input("Enter: ") 


with open('logging.txt', 'w') as file:
    file.write("Your conversation log:")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):    
            file.write(f"AI: {message.content}\n\n")
    file.write("End of conversation")

print('Conversation saved to file logging.txt')
