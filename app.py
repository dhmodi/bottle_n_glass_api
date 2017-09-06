#!/usr/bin/env python

from __future__ import print_function
from future.standard_library import install_aliases

install_aliases()


import os
from flask import Flask, request, redirect, url_for, flash, session, render_template
from flask_session import Session
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.utils import plot_model
import cv2
import numpy as np
import gc
gc.collect()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
sess = Session()
sess.init_app(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def image():
    flash('Image Upload')
    return render_template('index.html', page_title='My Page!')


@app.route('/image', methods=['GET', 'POST'])
def upload_file():
    category = "Others"
    filename = ""
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('upload_file',
            #                         filename=filename))
            model = load_model('partRecognition/model/model.h5')
            #
            # model.compile(loss='binary_crossentropy',
            #               optimizer='rmsprop',
            #               metrics=['accuracy'])

            model.compile(loss='binary_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])

            #   print (file)
            img = cv2.imread(UPLOAD_FOLDER + "/" + filename)
            img = cv2.resize(img, (224, 224))
            img = np.reshape(img, [1, 224, 224, 3])

            classes = model.predict_classes(img)

            if (classes[0] == 0):
                category = "Glass"
            elif (classes[0] == 1):
                category = "Bottle"
   #
    return {
        "fileName": filename,
        "objectCategory": category,
        "source": "deloitte-image-analytics"
    }

if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    port = int(os.getenv('PORT', 5000))

    print("Starting app on port %d" % port)

    app.run(debug=True, port=port, host='0.0.0.0')