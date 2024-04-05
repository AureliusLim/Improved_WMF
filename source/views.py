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
    new_width = width - d_width
    new_height = height - d_height

    #and radius is always 15 In author's example
    radius = 15
    output_array = np.zeros_like(channel_array)
    histogram = np.zeros_like((256,256))
    
    temp_necklace = [[] for _ in range(256)]

    for cur_col in range(d_width, new_width):
        rTopMostRow = min(new_height-1, radius)
        rLeftMostCol = max(0, cur_col - radius)
        rRightMostCol = min(width - 1, cur_col + radius)
        for r_cur_row in range(0, rTopMostRow - 1):
            for r_cur_col in range (rLeftMostCol, rRightMostCol-1):
                if mat[r_cur_row][r_cur_col] == 0:
                    continue
                fval = channel_array[r_cur_row][r_cur_col]
                gval = weights[r_cur_row][r_cur_col]
                    
                if histogram[fval][gval] == 0 and gval:
                    temp_necklace[fval].append(gval)
                    #necklace_table(channel_array, weights)
                histogram[fval][gval] += 1
                #updateBCB(BCB[gval],BCBf,BCBb,gval,-1);
        for cur_row in range(d_height, new_height):
            '''
            Do weighted median filtering here with the use of BCB
            '''
            #bcb finds the median and occurs everytime when the histogram shifts
            neighborhood = channel_array[y-d_height:y+d_height+1, x-d_width:x+d_width+1]
            median_value = np.median(neighborhood)
            output_array[cur_row, cur_col] = median_value

            #!!!!!update the joint histogram before going to the next row

            #1.)Insert new row pixel value count
            row_num = cur_row + radius + 1
            #if row index number is not past the image's border
            if(row_num < d_height):
                for r_cur_col in range (rLeftMostCol, rRightMostCol-1):
                    if mat[row_num][r_cur_col] == 0:
                        continue
                    fval = channel_array[row_num][r_cur_col]
                    gval = weights[row_num][r_cur_col]
                    
                    if histogram[fval][gval] == 0 and gval:
                        temp_necklace[fval].append(gval)
                        #necklace_table(channel_array, weights)
                    histogram[fval][gval] += 1
                    #updateBCB(BCB[gval],BCBf,BCBb,gval,-1);


            #2.)Remove bottommost row pixel value count
            row_num = cur_row - radius
            #if row index number is not past the image's border
            if(row_num >= 0):
                for r_cur_col in range (rLeftMostCol, rRightMostCol-1):
                    if mat[row_num][r_cur_col] == 0:
                        continue
                    fval = channel_array[row_num][r_cur_col]
                    gval = weights[row_num][r_cur_col]
                    histogram[fval][gval] -= 1
                    #remove the histogram from necklacke if no more pixel count
                    if histogram[fval][gval] == 0 and gval:
                        if gval in temp_necklace[fval]:
                            temp_necklace[fval].remove(gval)
                        #necklace_table(channel_array, weights)
                   
                    #updateBCB(BCB[gval],BCBf,BCBb,gval,-1);


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


        fast_runtime = 0
        #Improved weighted median filter
        '''
                img = Image.open(input_path)
                
        '''

        
        

        return render_template('startpage.html', uploaded_image=url_for('static', filename='uploads/' + filename), processed_image=url_for('static', filename='uploads/' + output_filename),  regular_runtime=regular_runtime, fast_runtime=fast_runtime)