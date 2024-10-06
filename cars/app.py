from flask import Flask, render_template

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

if __name__ == '__main__':
    app.run(debug=True)
