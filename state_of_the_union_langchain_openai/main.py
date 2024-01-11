import dotenv
import requests
import logging
import weaviate
from weaviate.embedded import EmbeddedOptions
    
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Weaviate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


dotenv.load_dotenv()


def chat(retriever, query: str):
    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
    """

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    if retriever is None:
        settings = {"question": RunnablePassthrough()}
        prompt = ChatPromptTemplate.from_template(template.replace("Context: {context}", ""))
    else:
        settings = {"context": retriever,  "question": RunnablePassthrough()} 
        prompt = ChatPromptTemplate.from_template(template)

    # LECL format: https://python.langchain.com/docs/expression_language/why
    rag_chain = (
        settings
        | prompt 
        | llm
        | StrOutputParser() 
    )

    result = rag_chain.invoke(query)
    logging.info(result)


def main_with_rag():
    # load data
    url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"
    res = requests.get(url)
    with open("state_of_the_union.txt", "w") as f:
        f.write(res.text)

    loader = TextLoader('./state_of_the_union.txt')
    documents = loader.load()

    # split documents into chunk
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # generate vector embedding
    client = weaviate.Client(embedded_options=EmbeddedOptions())

    vectorstore = Weaviate.from_documents(
        client = client,    
        documents = chunks,
        embedding = OpenAIEmbeddings(),
        by_text = False
    )
    retriever = vectorstore.as_retriever()

    # chat with RAG
    query = "What did the president say about Justice Breyer"
    chat(retriever, query)


def main_without_rag():
    query = "What did the president say about Justice Breyer"
    chat(retriever=None, query=query)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info('main_with_rag starts...')
    main_with_rag()
    logging.info('main_without_rag starts...')
    main_without_rag()
