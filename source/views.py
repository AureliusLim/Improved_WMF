from flask import Blueprint, render_template, request, current_app, url_for, redirect
from PIL import Image
import os
import numpy as np
import cv2
import time
from werkzeug.utils import secure_filename

main = Blueprint('main', __name__)


def weighted_median_filter(image, weights):
    
    if image.mode == 'RGB':
        
        r, g, b = image.split()
        
     
        r_filtered = weighted_median_filter_single_channel(np.array(r), weights)
        g_filtered = weighted_median_filter_single_channel(np.array(g), weights)
        b_filtered = weighted_median_filter_single_channel(np.array(b), weights)
        
       
        return Image.merge("RGB", (Image.fromarray(r_filtered), Image.fromarray(g_filtered), Image.fromarray(b_filtered)))
    else:
       
        img_array = np.array(image, dtype=np.uint8)
        filtered_array = weighted_median_filter_single_channel(img_array, weights)
        return Image.fromarray(filtered_array, mode='L')

def weighted_median_filter_single_channel(channel_array, weights):
    height, width = channel_array.shape
    weight_height, weight_width = weights.shape
    d_height, d_width = weight_height // 2, weight_width // 2
    output_array = np.zeros_like(channel_array)
    
    for y in range(d_height, height - d_height):
        for x in range(d_width, width - d_width):
            neighborhood = channel_array[y-d_height:y+d_height+1, x-d_width:x+d_width+1]
            median_value = np.median(neighborhood)
            output_array[y, x] = median_value

    brightening_factor = 1  
    output_array = np.clip(output_array * brightening_factor, 0, 255).astype(np.uint8)
    
    return output_array

def fast_weighted_median_filter(image, weights):
    pass


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
    
        img = Image.open(input_path)
        weights = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        regular_start_time = time.time()
        filtered_img = weighted_median_filter(img, weights)
        end_time = time.time()
        regular_runtime = end_time - regular_start_time
        print(regular_runtime)
        fast_runtime = 0
        output_filename = f'processed_{filename}'
        output_path = os.path.join(current_app.root_path, 'static/uploads', output_filename)
        filtered_img.save(output_path)
      
        
        

        return render_template('startpage.html', uploaded_image=url_for('static', filename='uploads/' + filename), processed_image=url_for('static', filename='uploads/' + output_filename),  regular_runtime=regular_runtime, fast_runtime=fast_runtime)