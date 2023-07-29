from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from dotenv import load_dotenv
import os


# Function to setup Pinecone API
def setup_pinecone(OPENAI_API_KEY):
    model_name = 'text-embedding-ada-002'
    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )
    YOUR_API_KEY = os.getenv("PINECONE_API_KEY")
    YOUR_ENV = os.getenv("PINECONE_ENV")
    index_name = 'langchain-retrieval-augmentation'
    pinecone.init(api_key=YOUR_API_KEY, environment=YOUR_ENV)
    text_field = "text"
    index = pinecone.Index(index_name)
    vectorstore = Pinecone(index, embed.embed_query, text_field)
    return vectorstore

# Function to query from OpenAI using Langchain
def query_openai(vectorstore, OPENAI_API_KEY, query):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo', temperature=0.0)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    return qa.run(query)

# Function to query with sources too
def query_with_sources(vectorstore, OPENAI_API_KEY, query):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo', temperature=0.0)
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    return qa_with_sources(query)

# Main function to run the script
def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    vectorstore = setup_pinecone(OPENAI_API_KEY)
    query = input("Input query: ")
    result = query_openai(vectorstore, OPENAI_API_KEY, query)
    print(result)
    #result_with_sources = query_with_sources(vectorstore, OPENAI_API_KEY, query)
    #print(result_with_sources)

# Run the main function when the script is executed directly
if __name__ == "__main__":
    main()
