####Tried apart from using the  ConversationalRetrievalChain to use the PromptTemplate
####ie in get_conversation_chain and get_conversation_chain_2 but some error ( llm Can't instantiate abstract class BaseLanguageModel with abstract methods agenerate_prompt) was coming by use GoogleGemini
####Instead Of FAISS (local) also tried to use PineCone (cloud)  as Vector DB to store Embeddings but Gemini Ai was not perfectly aligning with the dimensions on which Open AI aligns (1536 or 1538)


import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text




def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks




def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key='AIzaSyDugoEoAn72EAGwXMnmt2p6Ubi0OxfIQYs')
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore




def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    # llm=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3,google_api_key='AIzaSyDugoEoAn72EAGwXMnmt2p6Ubi0OxfIQYs')


    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain








# def get_conversational_chain_via_prompt():


#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in  the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n


#     Answer:
#     """


#     model = ChatGoogleGenerativeAI(model="gemini-pro",
#                              temperature=0.3,google_api_key='AIzaSyDugoEoAn72EAGwXMnmt2p6Ubi0OxfIQYs')


#     prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)


#     return chain






# def user_input(user_question,vectorstore):
#    
#     docs = vectorstore.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain(
#         {"input_documents":docs, "question": user_question}
#         , return_only_outputs=True)
#     print(response)










def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']


    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)




def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)


    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)


                # get the text chunks
                text_chunks = get_text_chunks(raw_text)


                # create vector store
                vectorstore = get_vectorstore(text_chunks)


                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)




if __name__ == '__main__':
    main()
