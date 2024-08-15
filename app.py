import os.path
import os
import streamlit as st
import pickle
from streamlit_extras.add_vertical_space import  add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain

with st.sidebar:
    st.title("LLM chat App")
    st.markdown(
        '''
        # About
        This App is an LLM-powered chatbot
        
        '''
    )
    add_vertical_space(5)
    st.write("Made by Hao Huang")


def main():
    st.header("Chat with PDF 😊")

    load_dotenv()

    pdf = st.file_uploader("Upload your PDF",type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)


        #embeddings

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
                # st.write("Embeddings load from the disk")
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore, f)
            # st.write("Embeddings Computation completed")

        # Accept user Question
        query = st.text_input("Ask questions about your PDF File")
        # st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm=llm,chain_type="stuff")
            response = chain.run(input_documents=docs,question=query)
            st.write(response)






if __name__ == '__main__':
    main()