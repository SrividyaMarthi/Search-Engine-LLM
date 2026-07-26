"""
Notes: uses wrappers as api calls, uses one search engine, saves memory in session state, agent uses llm and tools to get the answer from the web and required wrappers.
DIY: use with "integrate the pdf search chatbot with this and see"
"""

from langchain_core.tools.base import _handle_tool_error
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

#Arxiv  API wrapper
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max = 250)
arxiv = ArxivQueryRun(api_wrapper = api_wrapper_arxiv)

# WikipediaAPIWrapper--> Langchain wrapper uses wikipedia API to fetch info
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=4000,_handle_tool_error=True)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

search = DuckDuckGoSearchRun(name="Search")

st.title(" Langchain - Chat with search")
"""
Streamlitcallbackhandler: To display the thoughts and actions of the agents while interacting with the tools
"""


##Sidebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input(" Enter your GROQ API Key:", type="password")


if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, i'm chatbot who can search the web. How can i help you?"}
     ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
if prompt:=st.chat_input(placeholder = "What is Machine learning?"):
    st.session_state.messages.append({"role": "user","content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key = api_key, model_name = "meta-llama/llama-4-scout-17b-16e-instruct", streaming=True  )
    tools = [search,arxiv]

    #Converting tools to agents

    search_agent = initialize_agent(tools, llm, AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parse_errors =True)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        #response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        response =  search_agent.invoke({"input": prompt}, config={"callbacks": [st_cb]})
        st.session_state.messages.append({"role":'assistant', "content":response})
        st.write(response)