# spot detection fully based on RS-FISH
# requires fiji with RS-FISH installed to work

import os
import numpy as np
import pandas as pd
from skimage.io import imsave
from skimage.io import imread
import imageio
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from glob import glob
from natsort import natsorted
import json
from scipy.optimize import curve_fit

# creates the output folder if it doesn't yet exist
def create_folder(folder_path):
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        

# parses the RS-FISH config file for detection parameters                
def read_parameters(settings_file_path):
    
    # Read the settings file and extract the parameters
    parameters = {}
    with open(settings_file_path, "r") as settings_file:
        lines = settings_file.readlines()
        
        for line in lines:
            key_value = line.strip().split(" : ")
            
            if len(key_value) == 2:
                key = key_value[0].strip()
                value = key_value[1].strip()
                parameters[key] = value
                
    return(parameters)
                

# creates a fiji command for the RS-FISH, based on the parameters in the config file    
def make_fiji_command(images_path,out_path,settings_file_path,macro_path,fiji_path,channel):
    
    parameters = read_parameters(settings_file_path)
    
    # Construct the Fiji command with the extracted parameters
    fiji_command = (
        f"{fiji_path} --headless -macro "
        f"{macro_path} \""
        f"{images_path},"
        f"{out_path},"
#        f"{images_path.rsplit('/', 2)[0]}/detections/statsDetectionTime.txt,"
        f"{out_path}/statsDetectionTime.txt,"
        f"{parameters['anisotropyCoefficient']},"
        f"{parameters['RANSAC']},"
        f"{parameters['min intensity']},"
        f"{parameters['max intensity']},"
        f"{parameters['SigmaDoG']},"
        f"{parameters['ThresholdDoG']},"
        f"{parameters['supportRadius']},"
        f"{parameters['InlierRatio']},"
        f"{parameters['MaxError']},"
        f"{parameters['intensityThreshold']},"
        f"{parameters['bsMethod']},"
        f"{parameters['bsMaxError']},"
        f"{parameters['bsInlierRatio']},"
        f"{channel}\""
    )
    
    return(fiji_command)
    

# detect all spots in a imaged using RS-FISH, based on a sepcified detection config for each channel    
def detect_spots(images_path, detection_settings, channels,
                 tif_subfolder = "tif",
                 out_subfolder = "detections/",
                 macro_path = "/home/stumberger/fish-pipelines/fish_utils/RS_macro_param.ijm",
                 fiji_path = "/home/stumberger/tools/Fiji.app/ImageJ-linux64"):
    
    # tif path
    out_path = f"{images_path}/{out_subfolder}/"
    images_path = f"{images_path}/{tif_subfolder}/"
    
    create_folder(out_path)
    
    
    # process all channels
    for channel,settings_file in zip(channels,detection_settings):
    
        # read all parameters and pass them to fiji command
        fiji_command = make_fiji_command(images_path,out_path,settings_file,macro_path,fiji_path,channel)

        print(fiji_command)
        os.system(fiji_command)
        

# plots the spot detection on images, to check if the detection works    
def plot_detections(path, channel, path_spots=None, tif_subfolder="tif", out_folder=None, range_quantiles = (0.02, 0.9999)):
    
    # either the path to the upper folder containing the "detections" folder with merge.csv or the direct path to the merge.csv
    try:
        if path_spots is None:
            spots = pd.read_csv(f"{path}/{out_folder}/merge.csv")
        else:
            spots = pd.read_csv(path_spots)
    except FileNotFoundError:
        raise ValueError("Please provide a valid .csv file with spot information.")
        
    # create out folder if non-existant
    if out_folder == None:
        out = f"{path}/detections/vis/"
    else:
        out = out_folder
    create_folder(out)
    
    # get list of images
    tifs = glob(f"{path}/{tif_subfolder}/*.tif")
    tifs = [os.path.normpath(filepath) for filepath in tifs] # remove double //
    tifs = natsorted(tifs)
    
    # filter list for channels to visualise
    ch = ["_ch" + str(num) + "." for num in channel]
    tifs = [path for path in tifs if any(item in path for item in ch)]
    
    # iterate over all images in path
    for img in tifs:
        current_spots = spots[spots['img'] == img] # get all spots for image
        
        img1 = imread(img).max(axis=0)
        intensity_range = tuple(np.quantile(img1, range_quantiles))
        img_norm = rescale_intensity(img1, in_range=intensity_range, out_range='uint8').astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # plot side by side
        fig.suptitle(os.path.basename(img).rsplit(".", 1)[0])
        plt.subplots_adjust(top=1)
        axes[0].imshow(img_norm)
        axes[1].imshow(img_norm)

        for _, row in current_spots.iterrows():
            x, y, intensity = row['x'], row['y'], row['intensity']
            c = plt.Circle((x,y),7, edgecolor='r', facecolor = 'None')
            axes[1].add_patch(c)
            
        # save image
        fig.savefig(f"{out}/{os.path.basename(img)}.png",dpi=300)
        
        plt.close(fig)

# combines all spot csvs from all images
def combine_csv(path,tif_subfolder,out_subpath):

    folder_path = f"{path}/{out_subpath}/"
    files = glob(f"{folder_path}*.csv")
    files = natsorted(files)

    dataframes = []

    for file_name in files:
        
        try:
            
            # skip merge files to prevent self-merging
            if "merge" in file_name:
                continue
            
            df = pd.read_csv(file_name)

            # Extract the image name from the file name
            img = os.path.normpath(f"{path}/{tif_subfolder}/{file_name.split('_results_', 1)[1].split('_aniso', 1)[0]}")

            # Extract the channel number from the file name
            channel = int(file_name.split('_ch', 1)[1].split('.tif', 1)[0])

            # Add the 'img name' and 'channel' columns to the DataFrame
            df.insert(0, 'img', img)
            df.insert(1, 'channel', channel)

            dataframes.append(df)

        # exception for empty datafarmes
        except: 
                pass

    # Merge all DataFrames into a single DataFrame and save
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # add spot number
    merged_df['spot_idx'] = merged_df.groupby(['img', 'channel']).cumcount() + 1
    merged_df.to_csv(f"{folder_path}/merge.csv", index=False)

    return(merged_df)    
        
        
# add acquisition info to spots
def add_sample_info(path,out_subfolder="detections",info=None):
    
    spots = pd.read_csv(f"{path}/{out_subfolder}/merge.csv")
    
    # get metadata
    if info == None:
        info = f"{path}/acquisition_info.json"
    else:
        pass
    
    with open(info, 'r') as file:
        metadata = json.load(file)
        
    metadata = pd.concat([pd.json_normalize(metadata.get(key, {})).add_prefix(f'{key}.') for key in ['experiment', 'preparation', 'acquisition']], axis=1)    
    
    channels = metadata['acquisition.channels'][0]
    
    # expand columns with multiple entries (eg several channels)
    columns_to_explode = [col for col in metadata.columns if any(isinstance(item, list) for item in metadata[col])]
    metadata = metadata.explode(columns_to_explode)
    
    # unfiy channel names in acquisition info and the spots df
    channel_mapping = dict(zip(channels, list(range(0,len(channels)))))
    metadata['acquisition.channels'] = metadata['acquisition.channels'].map(channel_mapping)
    
    
    # combine metadata with spots
    df = spots.merge(metadata, right_on="acquisition.channels", left_on="channel", how='left')
    
    df.to_csv(f"{path}/{out_subfolder}/merge.csv", index=False)

# refine subpixel localization
def gaussian_3d(coords, x0, y0, z0, sigma_x, sigma_y, sigma_z, A, B):
    x, y, z = coords
    return (A * np.exp(
        -(((x - x0)**2) / (2 * sigma_x**2)
        + ((y - y0)**2) / (2 * sigma_y**2)
        + ((z - z0)**2) / (2 * sigma_z**2))
    ) + B).ravel()

def refine_subpixel(in_path, in_file,out_path,roi_radius = 5):
    # --- Load data ---
    spots_df_all = pd.read_csv(f'{in_path}/{in_file}')  # Columns: x, y, z
    
    refined_coords = []
    
    for img_path in spots_df_all['img'].unique():
    
        # Read image
        image = imread(img_path)
    
        # Subset df
        spots_df = spots_df_all[spots_df_all['img'] == img_path]
    
        # --- Loop through spots ---
        for idx, row in spots_df.iterrows():
            
            x0, y0, z0 = row['x'], row['y'], row['z']
            x0i, y0i, z0i = int(round(x0)), int(round(y0)), int(round(z0))
        
            # Crop ROI
            zmin, zmax = z0i - roi_radius, z0i + roi_radius + 1
            ymin, ymax = y0i - roi_radius, y0i + roi_radius + 1
            xmin, xmax = x0i - roi_radius, x0i + roi_radius + 1
        
            if (zmin < 0 or ymin < 0 or xmin < 0 or 
                zmax > image.shape[0] or ymax > image.shape[1] or xmax > image.shape[2]):
                continue  # Skip boundary cases
        
            roi = image[zmin:zmax, ymin:ymax, xmin:xmax]
        
            # Generate coordinate grid
            z_range, y_range, x_range = np.mgrid[
                zmin:zmax,
                ymin:ymax,
                xmin:xmax
            ]
            
            # Flatten for curve fitting
            coords = (x_range, y_range, z_range)
            roi_flat = roi.ravel()
        
            # Initial guess
            guess = (x0, y0, z0, 1.0, 1.0, 1.5, np.max(roi), np.min(roi))
        
            try:
                # gaussian fit
                popt, _ = curve_fit(gaussian_3d, coords, roi_flat, p0=guess)
                
                # Copy full original row and update it
                refined_row = row.copy()
                refined_row['x'] = popt[0]
                refined_row['y'] = popt[1]
                refined_row['z'] = popt[2]
                refined_row['sigma_x'] = popt[3]
                refined_row['sigma_y'] = popt[4]
                refined_row['sigma_z'] = popt[5]
                refined_row['amplitude'] = popt[6]
                refined_row['background'] = popt[7]
    
                refined_coords.append(refined_row)
                
            except RuntimeError:
                continue  # Fit failed
    
    
    # add new spots to dataframe
    refined_df = pd.DataFrame(refined_coords)
    refined_df.to_csv(f"{in_path}/{out_path}", index=False)
