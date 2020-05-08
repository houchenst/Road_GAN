
from scrape_roads import StreetMask
import numpy as np
import rasterio
import cv2
import os
import json
import math
from scipy.stats import ttest_ind_from_stats
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

save_dir = "./results/road_width/"
im1dir = "./data/full_data/rhode_island_data/20200407_152655_1105/"
im1 = "20200407_152655_1105_3B_AnalyticMS.tif"
im2dir = "./data/full_data/rhode_island_data/20200407_152647_1105/"
im2="20200407_152647_1105_3B_AnalyticMS.tif"
im3dir = "./data/full_data/rhode_island_data/20200407_152654_1105/"
im3 = "20200407_152654_1105_3B_AnalyticMS.tif"

SOBEL_X = np.array([[1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]])
SOBEL_Y = np.array([[1, 2, 1],
                  [0, 0, 0],
                  [-1,-2,-1]])

def show_image(title, image):
    height, width = image.shape[0], image.shape[1]
    # optimize for my laptop screen
    if height/width > 10/15:
        y = 1000
        x = width * y/height
    else:
        x = 1500
        y = height * x/width
    new_image = cv2.resize(image, (int(x),int(y)))
    cv2.imshow(title, new_image)
    cv2.waitKey(0)

def variance(samples):
    '''
    Compute the mean of the samples and the 
    variance
    '''
    mean = np.mean(samples, axis=0)
    total = 0
    for x in range(samples.shape[0]):
        total += np.sum(np.square(samples[x]-mean))
    return total/samples.shape[0], mean

def mean_deviation(samples, mean):
    '''
    Measured from the average variance from a 
    provided mean, not the sample mean
    '''
    total = 0
    for x in range(samples.shape[0]):
        total += np.sum(np.square(mean-samples[x]))
    return total/samples.shape[0]

def scene_gradients(raw, x_filt=SOBEL_X, y_filt=SOBEL_Y, channels=4):
    '''
    Calculates the x and y gradients for each channel in the image
    '''
    gxr = np.zeros(raw.shape)
    gyr = np.zeros(raw.shape)
    for c in range(channels):
        gxr[:,:,c] = convolve2d(raw[:,:,c], x_filt, mode="same")
        gyr[:,:,c] = convolve2d(raw[:,:,c], y_filt, mode="same")
    return gxr, gyr

def average_component_gradient(final_mask, marginal_mask, gxr, gyr, x_filt=SOBEL_X, y_filt=SOBEL_Y):
    '''
    This method calculates the gradient of the raw image in the x and y
    directions, and then calculates the component of the gradient that is
    projected onto a vector orthogonal to the edge of the road mask.
    params:
        final_mask     - the total road mask
        marginal mask  - the marginal change in the road mask
        gxr            - the raw image x gradients
        gyr            - the raw image y gradients
        x_filt         - convolutional filter used to detect horizontal gradients
        y_filt         - convolutional filter used to detect vertical gradients
    '''
    gxm = convolve2d(final_mask, x_filt, mode="same")
    gym = convolve2d(final_mask, y_filt, mode="same")
    # show_image("gxm", gxm)
    # show_image("gym", gym)


    if len(gxr.shape) == 2:
        numerator = np.multiply(gxr, gxm) + np.multiply(gyr, gym)
    else:
        channels = gxr.shape[2]
        numerator = np.zeros(gxr.shape)
        for c in range(channels):
            # comp_raw^mask = abs((raw DOT mask)/(MAG raw))
            numerator[:,:,c] = np.multiply(gxr[:,:,c], gxm) + np.multiply(gyr[:,:,c], gym)
    
    denominator = np.sqrt(np.square(gxr) + np.square(gyr))
    grad_comp = np.abs(np.divide(numerator,denominator))
    avg_grad_comp = np.mean(grad_comp, axis=2)
    avg_marginal_grad = np.mean(grad_comp[marginal_mask == 1])
    return avg_marginal_grad

def width_max_gradient(sm, key, gxr, gyr, image_mask, min_width=1, max_width=12):
    '''
    Returns the width that has the greatest average perpendicular gradient. This
    is the most likely average road width.
    :param sm: a StreetMask object containing a loaded Planet image
    :param key: the type of road to find the width of
    :param image_mask: a binary mask of what parts of the Planet scene were imaged
    :param gxr: the x gradients of the Planet scene
    :param gxy: the y gradients of the Planet scene
    :min_width: one less than the minimum width (pixels) to test for road width
    :max_width: the maximum width (pixels) to test for road width
    '''
    print(key)
    first = True
    ws = []
    gs = []
    use_keys = [key]
    for width in range(min_width,max_width):
        if not first:
            prev_mask = final_mask
        widths = [width]
        sm.draw_mask(use_keys, widths, save=False, show=False, save_to=save_dir)
        road_mask = sm.get_mask()/255.
        # combine road mask and not imaged mask
        final_mask = np.multiply(image_mask, road_mask)
        if not first:
            # marginal mask is the new pixels with the increased width
            marginal_mask = final_mask-prev_mask
            av_mar_grad = average_component_gradient(final_mask, marginal_mask, gxr, gyr)
            # show_image("Marginal Mask", marginal_mask)
            print(f'\tWidth: {width} ==> Gradient: {av_mar_grad}')
            ws.append(width)
            gs.append(av_mar_grad)
            # show_image("Planet Image", np.multiply(np.mean(planet_data, axis=-1),mask))
            # show_image("Final Mask", final_mask)
        first = False
    # now find the argmax (width) of the average gradients
    max_width = 0
    max_grad = -1 #all the gradients should be positive, so this should get reset
    for i in range(len(ws)):
        if gs[i] > max_grad:
            max_grad = gs[i]
            max_width = ws[i]
    return max_width

def all_widths(img, img_dir, save_to, keys):
    '''
    Saves a dictionary with the best road width (pixels) for each
    road type in keys for a given Planet image
    :param img: the filename (not full path) to the image
    :param img_dir: the directory path that the image is in
    :param save_to: a director to save the json output to
    :param keys: the keys to find the optimal road width for
    '''
    opt_width = {}
    with rasterio.open(os.path.join(img_dir,img)) as dataset:
        sm = StreetMask(dataset, img_dir, img.split(".")[0])
        sm.load_from_json()
        planet_data = dataset.read()
        # convert from band first data to band last data
        planet_data = np.swapaxes(planet_data, 0,1)
        planet_data = np.swapaxes(planet_data, 1,2)
        # mask roads outside of imaged area
        not_imaged_mask = np.copy(planet_data)
        not_imaged_mask = np.mean(not_imaged_mask, axis=-1)
        not_imaged_mask[not_imaged_mask > 0] = 1.0
        # janky noramalization
        planet_data = planet_data/np.max(planet_data)
        gxr,gyr = scene_gradients(planet_data)
        
        for key in keys:
            opt_width[key] = width_max_gradient(sm, key, gxr, gyr, not_imaged_mask)
    
    output_filename = img.split(".")[0] + "_optwidths.json"
    output_path = os.path.join(save_to, output_filename)
    output_fileobj = open(output_path, "w+")
    output_fileobj.write(json.dumps(opt_width))
    output_fileobj.close()
    

if __name__ == "__main__":
    keys = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street", "service", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"]
    # keys = ["residential"]
    all_widths(im2, im2dir, save_dir, keys)

    
            
            

    
