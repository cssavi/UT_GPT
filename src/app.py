import os
import logging
from flask import Flask, render_template, request
from utgpt import setup_pinecone, initialize_openai, query_with_sources
from dotenv import load_dotenv


template_dir = os.path.abspath('templates')
app = Flask(__name__, template_folder=template_dir, static_folder='../static')

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
vectorstore = setup_pinecone(OPENAI_API_KEY)
agent = initialize_openai(vectorstore, OPENAI_API_KEY)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    try:
        if request.method == 'POST':
            query = request.form.get('query')
            result = agent(query)['output']
    except Exception as e:
        logging.error(f"An error occurred while processing the query: {str(e)}")
        result = "An error occurred while processing your query. Please try again later."
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
