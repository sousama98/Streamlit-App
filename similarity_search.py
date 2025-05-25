from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import google.generativeai as genai
from langchain_core.vectorstores import InMemoryVectorStore
import os


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
llm = genai.GenerativeModel("gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)

#Defining Prompt Templates
template = """Answer the question based on the context below. If the answer is not in the context, say "I don't know".
{Context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


#Creating vectorstore and embeddings

vectorstore = InMemoryVectorStore.from_texts(
    ["Harrison worked at Mphasis"],
    embedding=embeddings,
)

#Querying the vectorstore
query = "Did Harrison work at Mphasis?"
docs = vectorstore.similarity_search(query, top_k=1)
print(docs[0].page_content)

#querying the retriver
retriever = vectorstore.as_retriever()
docs = retriever.invoke(query,top_k=1)
print(docs[0].page_content)
