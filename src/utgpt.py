from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent
from dotenv import load_dotenv
import pinecone
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

# Function to initialize the agent to be queried from
def initialize_openai(vectorstore, OPENAI_API_KEY):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo', temperature=0.0)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    tools = [ Tool( name='Knowledge Base', func=qa.run, description=( 'use this tool for every query to get '
                    'more information and stories on the topic'))]
    conversational_memory = ConversationBufferWindowMemory( memory_key='chat_history', k=5, return_messages=True)
    agent = initialize_agent( agent='chat-conversational-react-description', tools=tools, llm=llm, verbose=True,
                            max_iterations=3, early_stopping_method='generate', memory=conversational_memory)
    sys_msg = """You are a well-informed guide of the University of Texas at Austin focused on helping students. 
                You are trained on information from the r/UTAustin reddit wiki and posts. Answer the user's questions 
               with as much structured and specific information on the topic as you can."""
    prompt = agent.agent.create_prompt( system_message=sys_msg, tools=tools)
    agent.agent.llm_chain.prompt = prompt
    return agent

# Function to query with sources too
def query_with_sources(vectorstore, OPENAI_API_KEY, query):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo', temperature=0.0)
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    return qa_with_sources(query)

# Main function to run the script for testing
def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    vectorstore = setup_pinecone(OPENAI_API_KEY)
    query = input("Input query: ")
    result = initialize_openai(vectorstore, OPENAI_API_KEY, query)
    print(result)

# Run the main function when the script is executed directly
if __name__ == "__main__":
    main()
