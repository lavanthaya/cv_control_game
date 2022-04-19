from flask import Flask, render_template
from control import run_controls

app = Flask(__name__)


@app.route('/controls')
def hello():
    run_controls()
    return 'CONTROL WINDOW STARTED'

@app.route('/game')
def index():
    return render_template('index.html')