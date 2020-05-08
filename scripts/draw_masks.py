'''
This module is used to draw road masks for all Planet scenes
in <full_scene_dir>
'''

import os
import json
import numpy as np
import rasterio
from scrape_roads import StreetMask

dataset = "rhode_island_data"
full_scene_dir = f'./data/full_data/{dataset}'

width_file = "20200407_152647_1105_3B_AnalyticMS_optwidths.json"
width_path = f'./results/road_width/{width_file}'

def make_roads(filename, folder, road_types, road_widths):
    '''
    Draws a road mask for <filename>
    :param filename: the name of the file to draw the roads for
    :param folder: subfolder that the file is in
    :param road_types: the type of roads to draw
    :param road_widths: the widths corresponding to each road type (in pixels)
    :returns: None
    '''
    with rasterio.open(os.path.join(full_scene_dir, folder, filename)) as dataset:
        sm = StreetMask(dataset, full_scene_dir, filename.split(".")[0])
        sm.draw_mask(road_types, road_widths, save_to=full_scene_dir)

if __name__ == "__main__":
    # load in the width dictionary
    width_fo = open(width_path, "r")
    width_dict = json.loads(width_fo.read())
    width_file.close(width_fo)

    road_types = []
    road_widths = []

    # make list of road types and widths in the same order
    for road in width_dict:
        road_types.append(road)
        road_widths.append(width_dict[road])

    # iterate through all the files in the path
    for fldr in os.listdir(full_scene_dir):
        if os.path.isdir(fldr):
            for f in fldr:
                # make sure it's the right file type
                if f.endswith("AnalyticMS.tif"):
                    make_roads(f, fldr, road_types, road_widths)


    


