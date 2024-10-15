from flask import Flask, render_template, request, jsonify
from chatbot.chatbot import get_chatbot_response
from pathlib import Path


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/troubleshooting')
def troubleshooting():
    return render_template('pages/troubleshooting.html')

@app.route('/chatbot')
def chatbot():
    return render_template('pages/chatbot.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json['message']
    chat_history = request.json.get('history', [])
    print(chat_history)
    knowledge = open(Path(__file__).parent / 'knowledge/knowledge.md', 'r').read()
    # Get bot response
    bot_response = get_chatbot_response(
        user_message, 
        knowledge=knowledge,
        history=chat_history,
    )
    
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
