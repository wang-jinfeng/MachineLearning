import os

from flask import Flask, request

app = Flask(__name__)


@app.route("/userdesk", methods=['POST', 'GET'])
def userdesk():
    if request.method == "POST":
        print 'call post now'
        id = request.args.get('id')
        data = request.args.get('data')
        print id
        print data
        status = os.system('echo ' + id + " " + data)
        return str(status)

    if request.method == "GET":
        print 'call get now'
        id = request.args.get('id')
        data = request.args.get('data')
        print id
        print data
        status = os.system('echo ' + id + " " + data)
        return str(status)


if __name__ == '__main__':
    app.run()
