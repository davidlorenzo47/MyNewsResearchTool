import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (our openai api key)

st.title("News Research Tool ")
st.sidebar.title("Articles URLs")

urls = []
for i in range (3): #range is 3 since we will pass 3 articles
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

main_placeholder = st.empty()   #to show interaction in main page. Creating empty UI element.
llm = OpenAI(temperature=0.9, max_tokens=500)

url_btn_clicked = st.sidebar.button("Process URL")
file_path = "faiss_index.pkl"

if url_btn_clicked:
    # loading data (Step 1)
    loader = UnstructuredURLLoader(urls=urls)   #gettig all the URLs
    main_placeholder.text("Loading Data Step (1/3)...✅✅✅")    #UI
    data = loader.load()

    # splitting data (Step 2)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Splitting Text Step (2/3)...✅✅✅")   #UI
    docs = text_splitter.split_documents(data)

    # creating embeddings and save it to FAISS index (Step 3)
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Step (3/3)...✅✅✅") #UI
    time.sleep(2)

    # Saving the FAISS index to a pickle file in our memory/disk.
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Enter your Question: ")
if query:
    if os.path.exists(file_path):   #if file/FAISS Vector DB exists.
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)