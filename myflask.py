from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from scipy.sparse.linalg._eigen.arpack._arpack import debug
from build_vocab import Vocabulary
from sample import runn
app = Flask(__name__)
CORS(app)
@app.route('/upload', methods=['POST'])
def upload():
    image = request.files['file']
    image = Image.open(image)
    strr = runn(image)
    return {"filename": strr}
if __name__ == '__main__':
    app.run(debug=True)