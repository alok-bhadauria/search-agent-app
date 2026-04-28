import os
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from PyPDF2 import PdfReader

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

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
    description="Load and search content from https://www.nasa.gov/mission/artemis-iv/ when asked about artemis"
)

def _pdf_reader(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return f"Tool Used: PDF Reader\n{text[:1000]}"

pdf_reader = Tool(
    func=_pdf_reader,
    name="pdf_reader",
    description="Read the PDF file named 'ai-evolution.pdf' located in the same directory when user asks about PDF or document"
)

def _wikipedia_search(query: str) -> str:
    result = wikipedia.run(query)
    return f"Tool Used: Wikipedia\n{result}"

wikipedia_search = Tool(
    func=_wikipedia_search,
    name="wikipedia_search",
    description="Search Wikipedia"
)

def _arxiv_search(query: str) -> str:
    result = arxiv.run(query)
    return f"Tool Used: Arxiv\n{result}"

arxiv_search = Tool(
    func=_arxiv_search,
    name="arxiv_search",
    description="Search research papers from Arxiv"
)

def _open_search(query: str) -> str:
    result = search.run(query)
    return f"Tool Used: OpenSearch\n{result}"

open_search = Tool(
    func=_open_search,
    name="open_search",
    description="Search the web using search engine"
)

agent = create_agent(
    model=ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY")
    ),
    tools=[web_loader, pdf_reader, wikipedia_search, arxiv_search, open_search],
    system_prompt="You are a smart search assistant. Use the best tool and always mention which tool was used in the final answer."
)

st.title("Search Agent Application")

query = st.text_input("Enter your query:")

if st.button("Search"):
    if query:
        response = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        })
        text = response["messages"][-1].content
        clean_text = text.replace("**", "")
        st.write(clean_text)
