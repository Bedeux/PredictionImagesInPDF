import os
from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = "supersecretkey"

# ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save("/home/bbordenave/personalProjects/PredictionImagesInPDF/1. Data/temp PDF/"+filename)
            flash('File successfully uploaded')
            return redirect(url_for('display_files'))
        else :
            print("FILES NOT ALLOWED")
    return render_template('upload.html')

@app.route('/display')
def display_files():
    image_files = []
    text_files = []

    image_path = os.path.join("static", "image")
    text_path = os.path.join("static", "texte")

    for file in os.listdir(image_path):
        if file.lower().endswith(('.jpg', '.jpeg')):
            image_files.append(file)

    for file in os.listdir(text_path):
        if file.lower().endswith(('.jpg', '.jpeg')):
            text_files.append(file)

    return render_template('display_files.html', image_files=image_files, text_files=text_files)

if __name__ == '__main__':
    app.run(debug=True)
