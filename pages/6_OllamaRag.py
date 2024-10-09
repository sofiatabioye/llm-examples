import streamlit as st
from langchain.chains import create_history_aware_retriever
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv
from langchain import hub

rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
load_dotenv()
# Initialize Pinecone
# embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]);


# pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="YOUR_PINECONE_ENVIRONMENT")
# index_name = "llama-chat-index"
PINCECONE_INDEX_NAME = "smpc"
# Create Pinecone vector store
# embeddings = OpenAIEmbeddings()
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]);
vectorStore = PineconeVectorStore(index_name=PINCECONE_INDEX_NAME, 
                                  embedding=embeddings,
                                  pinecone_api_key=os.environ["PINECONE_API_KEY"])


# def main():
st.title("Chat with Ollama using LangChain and Pinecone")

# Set up the Ollama LLM
llm = OllamaLLM(model="llama3.1")
# conversation = ConversationalRetrievalChain(llm=llm, 
                                            # retriever=vectorStore.as_retriever())
# chain = rephrase_prompt | llm
conversation = create_history_aware_retriever(
llm, vectorStore.as_retriever(), rephrase_prompt)
# conversation.

# chain.invoke({"question": "What is LangChain?"})
# User input and chatbot response
user_input = st.text_input("You: ", "Hello!")
if user_input:
    response = conversation.invoke({"input": user_input, "chat_history": ''})
    st.write(f"Ollama: {response}")

# main()