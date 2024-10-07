from flask import Flask, render_template, request, jsonify
from chatbot.chatbot import get_chatbot_response

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
    bot_response = get_chatbot_response(user_message)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
