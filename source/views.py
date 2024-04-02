from flask import Blueprint, render_template, request, current_app, url_for, redirect
from PIL import Image
import os
from werkzeug.utils import secure_filename

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('startpage.html')

@main.route('/process_image', methods=['POST'])
def process_image():
    print(request.files)

   
    
    file = request.files['inputimage']
    if file.filename == '':
        print("Empty File")
        return redirect(url_for('main.home'))
    
    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(current_app.root_path, 'static/uploads', filename)
        file.save(input_path)
        print("Saving File")
        output_filename = f'processed_{filename}'
        output_path = os.path.join(current_app.root_path, 'static/uploads', output_filename)
      

        return render_template('startpage.html', uploaded_image=url_for('static', filename='uploads/' + filename), processed_image=url_for('static', filename='uploads/' + output_filename))