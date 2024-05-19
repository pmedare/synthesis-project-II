# main.py

import pandas as pd
import zipfile

from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
from datetime import datetime

import os



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('interface.html')

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'zip'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Save the file to the 'uploads' folder next to app.py
        filename = secure_filename(file.filename)
        filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filePath)
        
        response = {'message': 'File uploaded successfully! Wait for some seconds for your results.',
                    'filename': filename}

        # Perform data analysis on the uploaded CSV file
        perform_data_analysis(UPLOAD_FOLDER)

        # return redirect('http://127.0.0.1:5000//graph')
        return redirect(url_for('graph'))
        # return jsonify(response)
    else:
        return jsonify({'error': 'Invalid file format. Upload a .ZIP file containing a CSV for your edges and node classes.'})

@app.route('/graph/')
def graph():
    print("Loading graph...")
    return render_template('out.html')


def unzip_file(zip_file_path, extract_to):
    # Create a ZipFile object for the specified ZIP file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def perform_data_analysis(file_folder):
    # Unzip de file
    unzip_file(file_folder + '\\files.zip', file_folder)

    classesDF = pd.read_csv(file_folder + '\\elliptic_txs_classes.csv')
    edgesDF = pd.read_csv(file_folder + '\\elliptic_txs_edgelist.csv')
    return True


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
