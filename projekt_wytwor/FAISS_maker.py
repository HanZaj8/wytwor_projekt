#this part is not used in the initial chatbot application
#it shows how chatbot's FAISS was made
# 'pip install pypdf' and 'pip install faiss-cpu' is needed

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd

def reading_merged_pdf():
    loader = PyPDFLoader(folder_path + "\\materialy_dot_stresu.pdf")
    docs = loader.load_and_split()
    return docs

def making_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap  = 20,
        length_function = len,
    )
    chunks = text_splitter.split_documents(docs)

    return chunks

def creating_FAISS():
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2',
                                        model_kwargs={'trust_remote_code': True})
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store


folder_path = os.getcwd() 
docs = reading_merged_pdf() # exctracting Document objects list from pdf file (pages)
chunks = making_chunks(docs) # creating chunks
vector_store = creating_FAISS(chunks) #creating FAISS from chunks' embbedings