import os
from flask import Flask, render_template, request
from utgpt import setup_pinecone, query_openai, query_with_sources

template_dir = os.path.abspath('templates')
app = Flask(__name__, template_folder=template_dir)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    if request.method == 'POST':
        query = request.form.get('query')
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        vectorstore = setup_pinecone(OPENAI_API_KEY)
        result = query_openai(vectorstore, OPENAI_API_KEY, query)
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
