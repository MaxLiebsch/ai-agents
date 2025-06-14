from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import ToolMessage, HumanMessage, BaseMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, List, Sequence, Annotated
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_chroma import Chroma

import os


from dotenv import load_dotenv


load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

pdf_path = "leistungen_pflegeversicherung.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Pdf file not found: {pdf_path}")


pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"Pdf has {len(pages)} pages")
except Exception as e:
    print(f"Error loading pdf {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

page_split = text_splitter.split_documents(pages)

persistent_path = "./db"
collection_name = "leistungen-pflegeversicherung"

if not os.path.exists(persistent_path):
    os.makedirs(persistent_path)

    
try:
    vectorstore = Chroma.from_documents(
        documents=page_split,
        embedding=embeddings,
        persist_directory=persistent_path,
        collection_name=collection_name
    )
    print(f"Created ChromaDb vector store!")

except Exception as e:
    print(f"Vectorstore creation failed {e}")
    raise


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


@tool
def retriever_tool(query: str) -> str:
    """This function searches and returns the information from the Leistungen Pflegeversicherung document"""
    
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the document"
    
    results = []

    for i,doc in enumerate(docs):
        results.append(f"Document {i+1}: \n {doc.page_content}")
    
    return "\n\n".join(results)


tools = [retriever_tool]
llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState) -> AgentState:
    result = state["messages"][-1]

    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

system_prompt = SystemMessage(content="""
    You are an intelligent AI assitant to answer questions about "Leistungen in der Pflegeversicherung".
    Use the retriver tool available to answer the questions.
    Please always cite the specific parts of the documents you use in your answer.
""")

tool_dict = {our_tool.name: our_tool for our_tool in tools}

def llm_Agent(state: AgentState) -> AgentState:
    messages = list(state["messages"])
    messages = [system_prompt] + messages
    message = llm.invoke(messages)
    return {"messages":[message]}

def retriver_Agent(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls

    results = []

    for t in tool_calls:
        tool_name = t["name"]
        tool_args = t["args"]
        print(f"Calling tool {tool_name} with query: {tool_args.get('query', 'No query provided')}")

        if not t["name"] in tool_dict:
            print(f"\nTool: {tool_name}")
            result = f"Incorrect tool name, Please try and Select a valid to from the available tools."

        else: 
            result = tool_dict[tool_name].invoke(tool_args.get('query', ''))
            print(f"Result length: {len(str(result))}")

        results.append(ToolMessage(tool_call_id=t['id'],name=tool_name, content=str(result)))

    print("Tool Execution Complete. Back to the model!")

    return {"messages": results}


graph = StateGraph(AgentState)

graph.add_node('agent', llm_Agent)
graph.add_node('retriver', retriver_Agent)

graph.set_entry_point('agent')
graph.add_edge("retriver", "agent")

graph.add_conditional_edges("agent", should_continue, {
    True: "retriver",
    False: END
})

app = graph.compile()

with open("graph.png", "wb") as file:
    file.write(app.get_graph().draw_mermaid_png())

def running_agent():
    while True:
        user_input = input("Welche Frage haben Sie?")

        if user_input.lower() in ["exit", "quit"]:
            break

        messages = [HumanMessage(content=user_input)]

        result = app.invoke({"messages": messages})

        print(f"\n---- ANSWERS ----")
        print(result['messages'][-1].content)
        
running_agent()
