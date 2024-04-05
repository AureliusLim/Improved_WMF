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
    weights = weights / weights.sum()
    
    for y in range(d_height, height - d_height):
        for x in range(d_width, width - d_width):
            neighborhood = channel_array[y-d_height:y+d_height+1, x-d_width:x+d_width+1]
            weighted_list = []
            
        
            for i in range(weight_height):
                for j in range(weight_width):
                    val = neighborhood[i, j]
                    weight = weights[i, j]
                    
            
                    replication_factor = int(np.ceil(weight * 100))  
                    weighted_list.extend([val] * replication_factor)
            
            
            median_value = np.median(weighted_list)
            output_array[y, x] = median_value
    
    
    output_array = np.clip(output_array, 0, 255).astype(np.uint8)
    
    return output_array

def necklace_table(image, weights):
    whole_table = []
    not_empty = []
  
    # append indices of non-empty cells to list
    ctr = 0
    while ctr < len(whole_table):
        if ctr != None:
            not_empty.append(ctr)
        ctr += 1
      
    # data access, skipping empty cells
    for x in not_empty:
        print(whole_table[x])

    # insertion 
    head = not_empty[0]
    next = not_empty[1]

    # insert anywhere in between head and next element
    insert = random.randrange(head+1, next-1)
    whole_table.insert(insert, 'val')

    # inserted element is now the nearest to head
    next = insert

    # deletion
    # farthest non-empty cell gets deleted
    last = not_empty[-1]
    del whole_table[last]

def fast_weighted_median_filter(image, weights):
    mat  = np.zeros_like(img, dtype=np.uint8)
      
    if image.mode == 'RGB':
        
        r, g, b = image.split()

        r_filtered = fast_weighted_median_filter_single_channel(np.array(r), weights, mat)
        g_filtered = fast_weighted_median_filter_single_channel(np.array(g), weights, mat)
        b_filtered = fast_weighted_median_filter_single_channel(np.array(b), weights, mat)

        return Image.merge("RGB", (Image.fromarray(r_filtered), Image.fromarray(g_filtered), Image.fromarray(b_filtered)))
    else:
        img_array = np.array(image, dtype=np.uint8)
        filtered_array = weighted_median_filter_single_channel(img_array, weights)
        return Image.fromarray(filtered_array, mode='L')
    

def fast_weighted_median_filter_single_channel(channel_array, weights, mat):
    #	This doesnt acknowledge on what type of weight it wants (unweighted, guassian, Jacard)

    '''
    TODO: make the joint histogram, BCB, and Necklace Chain
    joint-histogram can, contrarily, regenerate weights every time the window shifts
    '''
    height, width = channel_array.shape
    weight_height, weight_width = weights.shape
    d_height, d_width = weight_height // 2, weight_width // 2
    
    #and radius is always 15 In author's example
    radius = 15
    output_array = np.zeros_like(channel_array)
    histogram = np.zeros_like((256,256))
    
    for y in range(d_height, height - d_height):
        rTopMostRow = min(height-1, radius)
        for x in range(d_width, width - d_width):
            rLeftMostCol = max(0, x - radius)
            rRightMostCol = min(width - 1, x + radius)

            #Joint histogram for loop (to initialize joint histogram) : I didnt consider the purpose of d_height/d_width here
            for r_cur_row in range(0, rTopMostRow - 1):
                for r_cur_col in range (rLeftMostCol, rRightMostCol-1):
                    if mat[r_cur_row][r_cur_col] == 0:
                        continue
                    fval = channel_array[r_cur_row][r_cur_col]
                    gval = weights[r_cur_row][r_cur_col]
                    
                    if histogram[fval][gval] == 0 and gval:
                        necklace_table(channel_array, weights)
                    histogram[fval][gval] += 1
                    #updateBCB(BCB[gval],BCBf,BCBb,gval,-1);

            
            #bcb finds the median and occurs everytime when the histogram shifts
            neighborhood = channel_array[y-d_height:y+d_height+1, x-d_width:x+d_width+1]
            median_value = np.median(neighborhood)
            output_array[y, x] = median_value

    brightening_factor = 1  
    output_array = np.clip(output_array * brightening_factor, 0, 255).astype(np.uint8)
    
    return output_array
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
        #Original weighted median filter
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
        output_filename = f'processed_{filename}'
        output_path = os.path.join(current_app.root_path, 'static/uploads', output_filename)
        filtered_img.save(output_path)


        #Improved weighted median filter
        '''
                img = Image.open(input_path)
                fast_runtime = 0
        '''

        
        

        return render_template('startpage.html', uploaded_image=url_for('static', filename='uploads/' + filename), processed_image=url_for('static', filename='uploads/' + output_filename),  regular_runtime=regular_runtime, fast_runtime=fast_runtime)