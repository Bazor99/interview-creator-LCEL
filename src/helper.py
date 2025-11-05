from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from operator import itemgetter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from src.prompt import *

load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def file_processor(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )

    texts = text_splitter.split_documents(data)

    page_text = [t.page_content for t in texts]
    doc = [Document(page_content = t) for t in page_text]

    return doc

def llm_pipeline(file_path):
    doc = file_processor(file_path)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=['text'])
    REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template,)
    
    chain1 = PROMPT_QUESTIONS | llm | StrOutputParser()
    chain2 = (
    {"existing_answer": chain1, "text": itemgetter("text")}
    | REFINE_PROMPT_QUESTIONS
    | llm
    | StrOutputParser())

    res = chain2.invoke({"existing_answer": "", "text": doc})

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(documents=doc, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})

    answer_template = PromptTemplate(
    input_variables=["context", "question"],
    template=answer_prompt,)

    res_list = [element for element in res if element.endswith('?') or element.endswith('.')]
    
    llm_chain = (
    {"context": retriever, "question":  RunnablePassthrough()}
    | answer_template
    | llm
    | StrOutputParser())
    return res_list, llm_chain