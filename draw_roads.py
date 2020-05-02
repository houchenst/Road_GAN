
from make_training_data import StreetMask
import numpy as np
import rasterio
import cv2
import os
import math
from scipy.stats import ttest_ind_from_stats
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

save_dir = "./results/road_width/"
im1dir = "./data/rhode_island_data/20200407_152655_1105/"
im1 = "20200407_152655_1105_3B_AnalyticMS.tif"
im2dir = "./data/rhode_island_data/20200407_152653_1105/"
im2="20200407_152653_1105_3B_AnalyticMS.tif"
im3dir = "./data/rhode_island_data/20200407_152654_1105/"
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


    

if __name__ == "__main__":
    # keys = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street", "service", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"]
    keys = ["residential"]
    # widths = 14*[3]

    with rasterio.open(os.path.join(im1dir,im1)) as dataset:
        sm = StreetMask(dataset, im1dir, im1.split(".")[0])
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


        first = True
        ws = []
        ts = []
        for width in range(1,15):
            if not first:
                # prev_mean = mean
                # prev_var = var
                prev_mask = final_mask
            widths = [width]
            sm.draw_mask(keys, widths, save=False, show=False, save_to=save_dir)
            road_mask = sm.get_mask()/255.
            # combine road mask and not imaged mask
            final_mask = np.multiply(not_imaged_mask, road_mask)
            # samples = planet_data[final_mask==1]
            # var, mean = variance(samples)
            # if first: 
            #     base_mean = mean
            #     base_var = var
            #     base_stdev = math.sqrt(var)
            #     base_num = samples.shape[0]
            if not first:
                # marginal mask is the new pixels with the increased width
                marginal_mask = final_mask-prev_mask
                # marginal_samples = planet_data[marginal_mask == 1]
                # deviation = mean_deviation(marginal_samples, base_mean)
                # stdev = math.sqrt(deviation)
                # num_samples = samples.shape[0]
                # t = np.sum(np.square(mean - base_mean)) / math.sqrt(var/num_samples + base_var/base_num)
                av_mar_grad = average_component_gradient(final_mask, marginal_mask, gxr, gyr)
                # prob = ttest_ind_from_stats(mean1=base_mean, std1=base_stdev, nobs1=base_num, mean2=mean, std2=stdev, nobs2=num_samples)
                # show_image("Marginal Mask", marginal_mask)
                print(f'Width: {width} ==> Gradient: {av_mar_grad}')
                # ws.append(width)
                # ts.append(t)
                # show_image("Planet Image", np.multiply(np.mean(planet_data, axis=-1),mask))
                # show_image("Final Mask", final_mask)
                # print(planet_data.shape)
                # print(planet_data[final_mask == 1])
            first = False
        # plt.scatter(ws, ts)
        # plt.show()
        # rand_samps = np.reshape(planet_data[1000:1010,1000:1010], (-1,4))
        # deviation = mean_deviation(rand_samps, base_mean)
        # stdev = math.sqrt(deviation)
        # num_devs = stdev/base_stdev
        # # show_image("Marginal Mask", marginal_mask)
        # print(f'Rands ==> Deviation: {num_devs}')


    
