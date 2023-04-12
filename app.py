import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma, Pinecone
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
import pinecone

# Set the path where you want to save the uploaded PDF file
SAVE_DIR = "pdf"


st.header('Question Answering with your PDF file')
st.write("Are you interested in chatting with your own documents, whether it is a text file, a PDF, or a website? LangChain makes it easy for you to do question answering with your documents.")
def qa(file, query, chain_type, k,api_key_pinecode,index_name,environment_pinecode):
    # load document
    loader = PyPDFLoader(file)
    #loader = UnstructuredPDFLoader(file)
    #loader = OnlinePDFLoader("https://wolfpaulus.com/wp-content/uploads/2017/05/field-guide-to-data-science.pdf")
    documents = loader.load()
    #print("doccs",documents)
    # split the documents into chunks
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # texts = text_splitter.split_documents(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    # initialize pinecone
    pinecone.init(
        api_key=api_key_pinecode, #"",  # find at app.pinecone.io
        environment=environment_pinecode #"northamerica-northeast1-gcp"  # next to api key in console
    )

    #index_name = "openaiindex"
    index_name = index_name
    #db = Chroma.from_documents(texts, embeddings)
    #db = Pinecone.from_texts(texts, embeddings)
    db = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chain to answer questions
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)
    result = qa({"query": query})
    print(result['result'])
    return result









with st.sidebar:
    st.header('Configurations')
    st.write("Enter OpenAI API key. This costs $. Set up billing at [OpenAI](https://platform.openai.com/account).")
    apikey = st.text_input("Enter your OpenAI API Key here",type="password")
    os.environ["OPENAI_API_KEY"] = apikey

    st.write("Enter Pinecode API key.  [Pinecode](https://www.pinecone.io/).")

    apikey2 = st.text_input("Enter your Pinecone Key here",type="password")

    #password = st.text_input("Enter a password", type="password")

    enviroment_pinecode = st.text_input("Enter your Pinecone your environment Key")

    index_name = st.text_input("enter index-name")





left_column, right_column = st.columns(2)
    # You can use a column just like st.sidebar:








with left_column:

    # Add a file uploader to the app
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Save the uploaded file to the specified directory
        file_path = os.path.join(SAVE_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File path {file_path}")
    query = st.text_input("enter your question")
    chain_type = st.selectbox(
        'chain type',
        ('stuff', 'map_reduce', "refine", "map_rerank"))
    k = st.slider('Number of relevant chunks', 1, 5)

    if st.button('Loading'):
        # Or even better, call Streamlit functions inside a "with" block:
        result=qa(file_path, query, chain_type, k, apikey2, index_name, enviroment_pinecode)




        with right_column:

            st.write("Output of your question")

            #st.write(result)

            #st.write(result['result'])
            st.subheader("Result")
            st.write(result['result'])

            st.subheader("source_documents")
            st.write(result['source_documents'][0])








