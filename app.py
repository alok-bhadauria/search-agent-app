import os
import streamlit as st
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

st.set_page_config(page_title="Search Agent", layout="wide")

st.sidebar.title("API Setup")
api_key = st.sidebar.text_input("Enter GROQ API Key", type="password")

if not api_key:
    st.title("Search Agent")
    st.info("Enter your GROQ API key in the sidebar to begin")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key

wikipedia = WikipediaAPIWrapper()
arxiv = ArxivAPIWrapper()
search = DuckDuckGoSearchRun()

def _web_loader(url: str) -> str:
    loader = WebBaseLoader(url)
    docs = loader.load()
    return f"Tool Used: WebBaseLoader\n{docs[0].page_content[:1000]}"

web_loader = Tool(
    func=_web_loader,
    name="web_loader",
    description="Load Artemis mission content from NASA website when asked about Artemis"
)

def _wikipedia_search(query: str) -> str:
    return f"Tool Used: Wikipedia\n{wikipedia.run(query)}"

def _arxiv_search(query: str) -> str:
    return f"Tool Used: Arxiv\n{arxiv.run(query)}"

def _open_search(query: str) -> str:
    return f"Tool Used: OpenSearch\n{search.run(query)}"

wikipedia_search = Tool(func=_wikipedia_search, name="wikipedia_search", description="Search Wikipedia")
arxiv_search = Tool(func=_arxiv_search, name="arxiv_search", description="Search research papers")
open_search = Tool(func=_open_search, name="open_search", description="Search the web")

agent = create_agent(
    model=ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key
    ),
    tools=[web_loader, wikipedia_search, arxiv_search, open_search],
    system_prompt="You are a smart search assistant. Always mention which tool was used."
)

st.title("🔎 AI Search Agent")

query = st.text_input("Ask anything")

if st.button("Search"):
    if query:
        with st.spinner("Searching..."):
            response = agent.invoke({
                "messages": [{"role": "user", "content": query}]
            })
            text = response["messages"][-1].content
            st.success("Result")
            st.write(text.replace("**", ""))
    else:
        st.warning("Enter a query to search")
