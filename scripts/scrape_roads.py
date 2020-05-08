import os
import json
import numpy as np
import cv2
import rasterio
from pyproj import Transformer
import overpy
from PIL import Image, ImageDraw
import math
import cProfile
import pstats
import progressbar
import argparse

WGS84 = "EPSG:4326"

def change_crs(input_crs, output_crs, x, y):
    '''
    Converts a set of coordinates between coordinate systems
    Not efficient for doing many transforms because it makes
    a new transformer each time
    '''
    t = Transformer.from_crs(input_crs, output_crs)
    return t.transform(x,y)

class StreetMask():
    '''
    Used to create and edit a map of streets in a given region
    '''
    def __init__(self, dataset, dir, name):
        # Data folder and image name
        self.dir = dir
        self.name = name
        # dataset and derivative fields
        self.crs = dataset.crs
        self.pix_to_crs = dataset.transform
        self.crs_to_pix = dataset.index
        self.nw = self.pix_to_crs*(0,0)
        self.se = self.pix_to_crs*(dataset.width, dataset.height)
        self.nw_wgs = change_crs(self.crs, WGS84, self.nw[0], self.nw[1])
        self.se_wgs = change_crs(self.crs, WGS84, self.se[0], self.se[1])
        # Vars related to OpenStreetMap
        self.api = overpy.Overpass()
        self.osm_street_tags = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street", "service", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"]
        # Vars for drawing
        self.mask = Image.new("L", (dataset.width, dataset.height))
        self.canvas = ImageDraw.Draw(self.mask)
        # Roads from OSM
        self.json_roads = {}

    def load_from_json(self):
        '''
        Loads road coordinates from an existing json file
        '''
        json_file_path = os.path.join(self.dir, self.name + "_roads.txt")
        json_file = open(json_file_path, "r")
        self.json_roads = json.loads(json_file.read())


    def save_to_json(self):
        '''
        Saves road coordinate dictionary to a json file
        '''
        json_file_path = os.path.join(self.dir, self.name + "_roads.txt")
        json_file = open(json_file_path, "w+")
        json_file.write(json.dumps(self.json_roads))
        json_file.close()


    def load_roads(self):
        '''
        Queries roads from OpenStreetMap and saves
        the results as json
        '''
        for tag in self.osm_street_tags:
            roads = self.osm_query(tag)
            if roads != []:
                self.json_roads[tag] = roads

    def get_mask(self):
        '''
        Returns road mask as np array
        '''
        return np.array(self.mask)

    
    def draw_mask(self, keys, widths, save=True, show=False, save_to=None):
        '''
        Uses results from OpenStreetMap to draw a 
        binary road mask for the image
        '''
        # use json to get road coordinates
        # keys = list(self.json_roads.keys())
        for i in range(len(keys)):
            tag = keys[i]
            width = widths[i]
            roads = self.json_roads[tag]
            for road in roads:
                flat_list = [coord for point in road for coord in point]
                self.canvas.line(flat_list, fill=255, width=width)
        if save:
            if save_to == None:
                self.mask.save(os.path.join(self.dir, self.name + "_roads.png"))
            else:
                self.mask.save(os.path.join(save_to, self.name + "_roads.png"))
        if show:
            self.mask.show()

    def pix_res(self,p1,p2):
        '''
        Returns the resolution (euclidean distance) of points based
        on their pixel location and the image resolution
        '''
        sat_res = 3
        euc_dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        res = sat_res*euc_dist
        # print(f'Resolution: {res} meters')

    def process_roads(self, roads):
        '''
        Takes in way objects from OSM and outputs
        series of pixel coordinates
        '''
        all_nodes = []
        splits = [0]
        print(f'\t{len(roads)} roads to process')
        reformatted_points=[]
        if len(roads) > 0:
            pbar = progressbar.ProgressBar()
            for road in pbar(roads):
            # for road in roads:
                success = False
                # Handle api failures by trying again
                while not success:
                    try:
                        nodes = road.get_nodes(resolve_missing=True)
                        success = True
                    except (overpy.exception.OverpassTooManyRequests, overpy.exception.OverpassGatewayTimeout) as e:
                        print(f'{e} ----> Trying again')
                if len(nodes) > 1:
                    for node in nodes:
                        all_nodes.append((node.lat, node.lon))
                    splits.append(splits[-1] + len(nodes))
            t = Transformer.from_crs(WGS84, self.crs)
            mod_nodes = list(t.itransform(all_nodes))
            np_nodes = np.asarray(mod_nodes)
            xs = list(np_nodes[:,0])
            ys = list(np_nodes[:,1])
            # rows, cols = rasterio.transform.rowcol(~self.pix_to_crs, xs, ys)
            # points = list(zip(cols, rows))
            points = []
            for i in range(len(xs)):
                c = self.crs_to_pix(xs[i], ys[i])
                points.append((c[1],c[0]))
            for x in range(len(splits)-1):
                road_slice = points[splits[x]:splits[x+1]]
                if len(road_slice) > 0:
                    reformatted_points.append(road_slice)
        return reformatted_points
        

    def osm_query(self, road_type):
        '''
        Queries the OpenStreetMap API for roads of a specific class and 
        returns a list of roads defined in pixel coordinates.
        Inputs:
            road_type- string, a openstreetmap road class
        Returns:
            A list of list of tuples. Each sublist is one road and each
            tuple is the pixel coordinates of one node on that road
        '''
        success = False
        # place to store all of the roads as image coords
        print(f"\tRoad type: {road_type}")
        # limit search to a (south, west, north, east) box
        bounding_box = (self.se_wgs[0], self.nw_wgs[1], self.nw_wgs[0], self.se_wgs[1])
        query_text = f'way["highway"="{road_type}"]{bounding_box};out;'
        # make request to openstreetmap
        while not success:
            try:
                results = self.api.query(query_text)
                success = True
            except (overpy.exception.OverpassTooManyRequests, overpy.exception.OverpassGatewayTimeout) as e:
                print(f'{e} ----> Trying again')
        roads = results.ways
        road_nodes = self.process_roads(roads)

        return road_nodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--clear_data", help="Deletes old jsons",
                        action="store_true")
    args = parser.parse_args()

    # change this to change what is being scraped
    data_dir = "./data/full_data/rhode_island_data"

    
    for sub_dir in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir,sub_dir)):
            for img in os.listdir(os.path.join(data_dir, sub_dir)):
                if img.endswith("AnalyticMS.tif"):
                    full_img_path = os.path.join(data_dir, sub_dir, img)
                    json_file_path = os.path.join(data_dir, sub_dir, img.split(".")[0]+"_roads.txt")
                    # open the geotiff using rasterio
                    with rasterio.open(full_img_path) as dataset:
                        processed = False
                        for f in os.listdir(os.path.join(data_dir, sub_dir)):
                            # check if the roads have already been scraped
                            if f.endswith("_roads.txt"):
                                print(f'{img} already processed')
                                # if the data is supposed to be deleted
                                if args.clear_data:
                                    # os.remove(os.path.join(data_dir, sub_dir, f))
                                    print("Removed")
                                existing_json = open(os.path.join(data_dir, sub_dir, f), "r")
                                print(f'\t{len(json.loads(existing_json.read()).keys())} road types extracted')
                                processed = True
                        if not processed:
                            print(img)
                            mask = StreetMask(dataset, os.path.join(data_dir, sub_dir), img.split(".")[0])
                            mask.load_roads()
                            json_file = open(json_file_path, "w+")
                            json_file.write(json.dumps(mask.json_roads))
                            json_file.close()
                        dataset.close()
        
                




        
