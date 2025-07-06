from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def compress(img_path, n_colors, output_npz_path):
    img = Image.open(img_path)
    img = np.array(img)
    height, width, channels = img.shape
    shape = np.array([height, width, channels], dtype=np.int32)
    pixels = img.reshape(-1, channels)

    km = KMeans(n_clusters=min(n_colors, 256 * 256), init='k-means++')
    km.fit(pixels)
    centroids = km.cluster_centers_.astype(np.uint8)
    labels = km.labels_

    if n_colors < 257:
        labels = labels.astype(np.int8) 
    else:
        labels = labels.astype(np.int16)

    np.savez_compressed(output_npz_path, labels=labels, centroids=centroids, shape=shape)

def extract(npz_path, output_img_path):
    data = np.load(npz_path)
    labels, centroids, shape = data['labels'], data['centroids'], data['shape']
    extracted_img = np.empty((shape[0] * shape[1], shape[2]), dtype=np.uint8)
    for ind, label in enumerate(labels):
        extracted_img[ind] = centroids[label]
    extracted_img = extracted_img.reshape(shape)
    extracted_img = extracted_img.astype(np.uint8)
    ex_img = Image.fromarray(extracted_img)
    ex_img.save(output_img_path)

@app.route('/')
def index():
    return render_template('index.html', results=[])

@app.route('/compress', methods=['POST'])
def handle_compression():
    files = request.files.getlist('image')
    n_colors = int(request.form['n_colors'])

    results = []

    for file in files:
        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)

            base_name = os.path.splitext(filename)[0]
            compressed_npz = os.path.join(app.config['PROCESSED_FOLDER'], base_name + '.npz')
            compressed_img = os.path.join(app.config['PROCESSED_FOLDER'], base_name + '_compressed.png')

            compress(input_path, n_colors, compressed_npz)
            extract(compressed_npz, compressed_img)

            results.append({
                'original': url_for('static', filename=f'uploads/{filename}'),
                'compressed': url_for('static', filename=f'processed/{base_name}_compressed.png'),
                'npz': url_for('static', filename=f'processed/{base_name}.npz')
            })

    return render_template('index.html', results=results)

@app.route('/extract', methods=['POST'])
def handle_extraction():
    npz_file = request.files['npz_file']
    if npz_file:
        filename = secure_filename(npz_file.filename)
        npz_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        npz_file.save(npz_path)

        base_name = os.path.splitext(filename)[0]
        output_img_path = os.path.join(app.config['PROCESSED_FOLDER'], base_name + '_extracted.png')
        extract(npz_path, output_img_path)

        return render_template('index.html', results=[{
            'original': None,
            'compressed': url_for('static', filename=f'processed/{base_name}_extracted.png'),
            'npz': url_for('static', filename=f'uploads/{filename}')
        }])
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
