from sear import search
from flask import Flask
from flask import url_for, render_template, request, redirect

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET'])
def res():

    query = request.args['words']
    search_method = request.args['model']

    result = search(query, search_method)
    return render_template('results.html', result=result)


if __name__ == '__main__':
    app.run(host='localhost', port=5898, debug=True)
