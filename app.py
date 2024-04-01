from flask import Flask, render_template, request, jsonify

import GPT_RAG_Response_Generator

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get')
def get_bot_response():
    message = request.args.get('msg')
    response = ""
    if message:
        gpt_answer, rag_answer = GPT_RAG_Response_Generator.gpt_rag_response(message)
        if gpt_answer == "none":
            response = rag_answer
            return response
        elif rag_answer == "none":
            response = gpt_answer
            return response
        elif gpt_answer and rag_answer:
            response = gpt_answer + " " + rag_answer
            return response


if __name__ == "__main__":
    app.run(debug=True)
