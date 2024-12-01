import streamlit as st
import validators
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
##
st.set_page_config(page_title="LANGCHAIN SUMMARISER")
st.title("Summarise text from YT or website")
st.subheader("Summarise the url")
##CODE

with st.sidebar:
    groq_api_key=st.text_input("Enter your Groq Api key:",type="password")
    
url=st.text_input("URL")
llm=ChatGroq(groq_api_key=groq_api_key,model="gemma2-9b-it",streaming=True)

prompt_template="""
Provide the summary of following content in 300 words:
content:{text}
"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarise the content from YT or website"):
    if not groq_api_key.strip() or not url.strip():
        st.error("please provide the information")
    elif not validators.url(url):
        st.error("invalid error")
    else:
        try:
            with st.spinner("waiting...."):
                if "youtube.com" in url:
                    loader=YoutubeLoader.from_youtube_url(url,add_video_info=False)
                else:
                    loader=UnstructuredURLLoader(urls=[url],ssl_verify=False)

                docs=loader.load()
                
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                
                response=chain.run(docs)
                st.success(response) 
        except Exception as e:
            st.exception(f"Exception:{e}")
