'''
This module is designed to split up Planet scens and their
corresponding masks into smaller images for training a Pix2Pix
architecture
'''

import os
import json
import numpy as np
import rasterio
from scrape_roads import StreetMask
from PIL import Image

TRAINING_WIDTH = 256
TRAINING_HEIGHT = 256

dataset = "rhode_island_data"
full_scene_dir = f'./data/full_data/{dataset}'
output_dir = f'./data/training_data/{dataset}'

def split_planet_scene(filename, folder):
    '''
    Splits one file and mask up into training examples
    '''

    img_dir = os.path.join(full_scene_dir, folder)
    roads_file = filename.split(".")[0] + "_roads.png"

    # load the road mask into an np array
    road_array = np.array(Image.open(os.path.join(full_scene_dir, folder, roads_file)))

    # use rasterio to load in the planet scene
    with rasterio.open(os.path.join(img_dir,filename)) as dataset:
        planet_data = dataset.read()
        
        scene_height = planet_data.shape[1]
        scene_width = planet_data.shape[2]


        # random sample to make roughly 80-10-10 train-validation-test split
        data_group = "train"
        sample = np.random.uniform()
        if sample > 0.9:
            data_group = "test"
        elif sample > 0.8:
            data_group = "validation"

        for i in range(planet_data.shape[0]+1):
            if i == 0:
                band = road_array
                label = "roadmask"
            else:
                band = planet_data[i-1]
                label = f'band{i}'

            for row_i in range(scene_height//TRAINING_HEIGHT):
                for col_i in range(scene_width//TRAINING_WIDTH):
                    section = band[row_i*TRAINING_HEIGHT:(row_i+1)TRAINING_HEIGHT, col_i*TRAINING_WIDTH:(col_i+1)*TRAINING_WIDTH]
                    section_name = f'{filename.split(".")[0]}_{row_i}_{col_i}_{label}.png'
                    section_dir = os.path.join(output_dir, data_group, label)

                    section_image = Image.fromarray(section)
                    section_image.save(os.path.join(section_dir, section_name))


if __name__ == "__main__":

    test_fldr = "data/full_data/rhode_island_data/20200407_131752_1054/"
    test_f = "20200407_131752_1054_3B_AnalyticMS.tif""

    split_planet_scene(test_f, test_fldr)

    # # iterate through all the files in the path
    # for fldr in os.listdir(full_scene_dir):
    #     fldr_path = os.path.join(full_scene_dir,fldr)
    #     if os.path.isdir(fldr_path):
    #         for f in os.listdir(fldr_path):
    #             # make sure it's the right file type
    #             if f.endswith("AnalyticMS.tif"):
    #                 split_planet_scene(f, fldr)



