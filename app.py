import os
from flask import Flask, request, render_template, send_from_directory

import torch
from torch_utils import predict

app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'GET':
            return render_template('home.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)

        classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
        result = predict(full_name)
        
        acc, cls = torch.max(result, dim=1)
        confidence = round(acc.item() * 100, 2)
        label = classes[cls.item()]

        return render_template('result.html', image_file_name = file.filename, label = label, confidence = confidence)


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
   app.run()
