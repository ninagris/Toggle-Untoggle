import numpy as np
import pandas as pd
import cv2
import os
import warnings

from PyQt6.QtGui import QPixmap, QImage
from PIL import Image
from skimage import measure, segmentation
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, disk

warnings.filterwarnings("ignore")

def delete_dot_underscore_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("._"):
                os.remove(os.path.join(root, file))
                print(f"Deleted: {os.path.join(root, file)}")

def open_folder(image_folder, ids_list):
    """
    Putting sets of TIFF images needed for segmentation in the dictionary
    """
    image_dict = {}
    delete_dot_underscore_files(image_folder)
    if not os.path.isdir(image_folder):
        return image_dict
    # Iterate through all files in the directory
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.tif', '.tiff')) and any(id in filename for id in ids_list):  # Extract all tif files
            file_path = os.path.join(image_folder, filename)
            # Add filename as key and image path as value to the dictionary
            image_dict[filename] = file_path
    return image_dict

def increase_contrast_stretch(image, low_percent, high_percent):
    """
    Increasing the contrast of the images for display in color
    """
    p_low = np.percentile(image, low_percent)
    p_high = np.percentile(image, high_percent)
    return np.clip((image - p_low) / (p_high - p_low), 0, 1)

def pixel_conversion(big_df, pixel_rate):
    """
    Pixel to micron conversion for the final df
    """
    area_rate = pixel_rate ** 2
    for col in ['area', 'bbox_area', 'area_convex']:
        if col in big_df:
            big_df[col] *= area_rate
    for col in ['perimeter', 'major_axis_length', 'minor_axis_length',
                    'equivalent_diameter_area', 'feret_diameter_max', 'perimeter_crofton']:
        if col in big_df:
            big_df[col] *= pixel_rate
    return big_df

def image_preprocessing(main_marker_image_path, # image of the channel that will be used for segmentation purposes (red/actin channel used here)
    nucleus_image_path, # image with the nuclei of the cells (blue/dapi channel)
    main_marker_channel = 'red', # can also be green
    main_marker_contrast_low = 15,
    main_marker_contrast_high = 99,
    nucleus_contrast_low = 15,
    nucleus_contrast_high = 99,
    min_non_black_pixels_percentage= 5,  # minimum percentage of the total image area to indicate the presence of objects
    intensity_threshold=62.0, # min intensity that considers pixel to be non-background
    pixel_conv_rate = 0.18,
    diam = 22):
    """
    Opening image and cleaning it up prior to segmentation
    """
    # Open the image file using PIL and convert it to an array
    main_marker_opened = Image.open(main_marker_image_path)
    main_marker_image = np.array(main_marker_opened)
    nucleus_opened = Image.open(nucleus_image_path)
    nucleus_image = np.array(nucleus_opened)
    # Increasing the contrast of the separate channel images
    main_marker_im = increase_contrast_stretch(main_marker_image, main_marker_contrast_low, main_marker_contrast_high) 
    nucleus_im = increase_contrast_stretch(nucleus_image, nucleus_contrast_low, nucleus_contrast_high)
    height, width = main_marker_im.shape
    rgb_image = np.zeros((height, width, 3))
    total_image_area = height * width

    # Count non-black pixels (see if the minimumm amount of pixels have the intensity above the threshold); skip the images if the image is empty
    non_black_pixel_count = np.sum(main_marker_image > intensity_threshold)
    if non_black_pixel_count < (min_non_black_pixels_percentage / 100) * total_image_area:  # converting the % specified by the user into a pixel number
        return None
    
    # Creating a merged image and adjusting parameters for the cellpose segmentation
    if main_marker_channel == 'red':
        marker_channel = 0
    elif main_marker_channel=='green':
        marker_channel = 1
    
    rgb_image[..., 2] = nucleus_im  # Blue channel for the nucleus
    rgb_image[..., marker_channel] = main_marker_im # Red channel for segmentation marker
    diam_pixels = diam / pixel_conv_rate # Converting microns specified by the user in pixels

    return main_marker_image, nucleus_image, diam_pixels, marker_channel, rgb_image


def analyze_segmented_cells(predicted_masks,
                         main_marker_image,
                         main_marker_image_name,
                         nucleus_image,
                         min_nucleus_pixels_percentage,
                         nucleus_pixel_threshold,
                         min_area,
                         pixel_conv_rate,
                         rgb_image,
                         condition_name,
                         replicate_num,
                         third_marker = None,
                         fourth_marker=None,
                         properties = ['label', 'area', 'bbox_area', 'area_convex', 'perimeter', 'eccentricity', 'extent', 'major_axis_length', 
                        'minor_axis_length', 'centroid', 'mean_intensity', 'max_intensity', 'min_intensity',
                        'equivalent_diameter_area', 'feret_diameter_max' ,'orientation','perimeter_crofton','solidity']):
    """
    Extracting morphological params from segmented cells
    """
    labeled_mask = measure.label(predicted_masks)
    cleared_mask = segmentation.clear_border(labeled_mask)
        
    valid_props = []
    valid_regions = []
    mask_list = []  # List that is eventually populated with dictionaries for individual objects
    new_label_counter = 1 # to make sure mask labeling starts with 1 after filtering
    for region in measure.regionprops(cleared_mask, intensity_image=main_marker_image): # Calculating properties for the masks of cells
        region_mask = (cleared_mask == region.label) # Obtaining the mask of the individual regions
        """ Checking for the presence of nucleus inside the cell"""
        region_area = region.area
        min_required_nucleus_pixels = (min_nucleus_pixels_percentage / 100) * region_area # Converting the % into pixel number
        nucleus_pixels = nucleus_image[region_mask] 
        region_nucleus_pixels = np.sum(nucleus_pixels >= nucleus_pixel_threshold)
                
        if region_nucleus_pixels >= min_required_nucleus_pixels and region_area > (min_area/pixel_conv_rate**2):
            props = {prop: getattr(region, prop) for prop in properties} 
            valid_props.append(props)
            valid_regions.append(region)
        
            # Appending dictionary with the info for the cell to the list for the whole image
            mask_list.append({
                "image_name": main_marker_image_name,
                "label": new_label_counter,
                "mask": region_mask
            })

            new_label_counter += 1  # Increment for next valid object
    
    # Generating grayscale image for further mask overlay
    rgb_image_copy = rgb_image.copy()
    if rgb_image_copy.dtype != np.uint8:
        rgb_image_copy = (rgb_image_copy * 255).astype(np.uint8)
    gray_image = cv2.cvtColor(rgb_image_copy, cv2.COLOR_BGR2GRAY) 
    overlay_image = rgb_image.copy()
    if overlay_image.dtype != np.uint8:
        overlay_image = (overlay_image * 255).astype(np.uint8)
   
    for region_data in mask_list:
        mask_data = region_data["mask"]
        image_size = max(rgb_image.shape[:2])  # Getting the larger dimension (height or width)
        scaling_factor = image_size / 1000  # Adjusting the divisor to control scaling
        thickness = max(1, int(5 * scaling_factor))  # Ensuring a minimum thickness of 1
        boundaries = find_boundaries(mask_data, mode='outer')  
        thick_boundaries = dilation(boundaries, disk(thickness))  
        overlay_image[thick_boundaries.astype(bool)] = [255, 255, 255]  

    if valid_props: 
        temp_df = pd.DataFrame(valid_props)
        # Overriding skimage-given labels to have them start at 1
        temp_df['label'] = list(range(1, len(temp_df) + 1))
        # Cleaning up centroid into two float columns
        temp_df[['centroid_y', 'centroid_x']] = pd.DataFrame(temp_df['centroid'].tolist(), index=temp_df.index)
        temp_df.drop(columns='centroid', inplace=True)

        temp_df['image_name'] = main_marker_image_name
        temp_df = temp_df[['image_name'] + [col for col in temp_df.columns if col != 'image_name']]
        
        temp_df['Condition']=condition_name
        temp_df['Replicate']=replicate_num
        
        return temp_df, overlay_image, gray_image, mask_list

def convert_to_pixmap(array, format):
    """
    Convert a numpy array to QPixmap to display in GUI
    """
    height, width = array.shape[:2]
    if array.ndim == 2:  # Grayscale image
        channels = 1
    elif array.ndim == 3 and array.shape[2] == 4:  # RGBA image
        channels = 4
    else:
        raise ValueError("Unsupported array shape for conversion to QPixmap.")
    bytes_per_line = width * channels
    q_image = QImage(array.data, width, height, bytes_per_line, format)
    return QPixmap.fromImage(q_image)


def normalize_to_uint8(array):
    """Normalize and convert array to uint8 for QImage conversion."""
    array = array - array.min()
    array = (array / array.max() * 255).astype(np.uint8)
    return array