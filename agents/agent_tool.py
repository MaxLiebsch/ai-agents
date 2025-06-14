from typing import TypedDict, Union, Sequence, Annotated
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
     messages: Annotated[Sequence[Union[BaseMessage, AIMessage,ToolMessage]], add_messages]

@tool
def add(a: int, b: int) -> int:
    """This is an addition function that adds two numbers together"""
    return a + b

@tool
def multiply(a:int, b:int) -> int:
    """Multiply function"""
    return a * b

@tool
def substract(a: int, b: int) -> int:
    """Substract function"""
    return a - b

tools = [add, multiply, substract]

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
     system_prompt = SystemMessage(
          content="You are my AI assistant. Please answer my queries to the best of your abilities"
     )
     response = llm.invoke([system_prompt] + state['messages'])
     return {"messages": [response]}



def should_continue(state: AgentState) -> AgentState:
     last_message = state["messages"][-1]
     if not last_message.tool_calls:
        return "end"
     else:
        return 'continue'
     

graph = StateGraph(AgentState)
graph.add_node('our_agent', model_call)

toolNode = ToolNode(tools=tools)
graph.add_node('tools', toolNode)

graph.add_conditional_edges("our_agent", should_continue, {
    "continue": "tools",
    "end": END
})

graph.add_edge('tools', "our_agent")

graph.set_entry_point('our_agent')

app = graph.compile()

with open("graph.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())


def print_stream(stream):
    for s in stream:
        message = s['messages'][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 30 + 12 and substract 10 from the result. Multiply 3 * 7")]}
print_stream(app.stream(inputs, stream_mode="values"))


