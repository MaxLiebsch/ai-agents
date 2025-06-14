from typing import TypedDict, Sequence, Annotated, Union
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[Union[BaseMessage, ToolMessage, AIMessage]], add_messages]

document_content = ""

@tool
def update(content: str) -> str:
    """Update the local content with the provided content."""
    global document_content
    document_content = content

    return f"Document has been updated successfully! The content is: \n{document_content}"

@tool
def save(filename: str):
    """Save the current document as a textfile and finish the process
    
    Args:
        filename: Name of the text file
    
    """

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n📄 Document has been saved to: {filename}")
        return f"Document has been saved successfully to {filename}"
    except Exception as e:
        return f"Error saving document {e}"


tools = [update, save]

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are a drafter, a helpful writing assistant. You are going to help the user update and save document.                              

    - if the user wants to update or modify a document, use the 'update' tool with the complete updated content.
    - if the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current content of the document after modifications.

    The current document content is: {document_content}""")
    
    if not state['messages']:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else: 
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages =  [system_prompt] + list(state['messages']) + [user_message]
    
    response = llm.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"\n🛠 TOOL: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state['messages']) + [user_message, response]}


def should_continue(state: AgentState) -> AgentState: 
    """Determine if whe should continue with updating the document or save it"""

    if not state['messages']:
        return 'continue'
    
    for message in reversed(state['messages']):
        if(isinstance(message, ToolMessage) and 
           "saved" in message.content.lower( ) and 
           'document' in message.content.lower()):
            return "end"
        
    return "continue"


def print_messages(messages):
    if not messages:
        return
    
    for message in messages[-3:]:
        if(isinstance(message, ToolMessage)):
            print(f"\n TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)

toolNode = ToolNode(tools)
graph.add_node('tools', toolNode)

graph.set_entry_point("agent")

graph.add_conditional_edges("tools", should_continue, {
    "continue": "agent"
    , "end": END
})

graph.add_edge("agent", "tools")

app = graph.compile()

with open("graph.png", "wb") as file:
    file.write(app.get_graph().draw_mermaid_png())


def run_document_agent():
    print(f"\n ++++++ DRAFTER ++++++")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n +++++ DRAFTER FINISHED +++++")

run_document_agent()
        