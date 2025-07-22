import numpy as np
import pandas as pd
import cv2
import os
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from skimage import measure, segmentation
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, disk
from PyQt6.QtGui import QPixmap, QImage


def delete_dot_underscore_files(directory):
    """
    Delete all hidden files starting with a dot (e.g., .DS_Store, ._filename) 
    from the given directory and its subdirectories.
    """
    for root, _ , files in os.walk(directory):
        for file in files:
            if file.startswith("."):
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
    # Iterating through all files in the directory
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.tif', '.tiff')) and any(id in filename for id in ids_list):  # Extracting all tif files
            file_path = os.path.join(image_folder, filename)
            # Adding a filename as key and an image path as value to the dictionary
            image_dict[filename] = file_path
    return image_dict


def increase_contrast_stretch(image, low_percent, high_percent):
    """
    Increasing the contrast of the images for display in color using clipping
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


def image_preprocessing(main_marker_image_path, 
    nucleus_image_path = None, 
    main_marker_channel = 'red', 
    main_marker_contrast_low = 15, 
    main_marker_contrast_high = 99,
    nucleus_contrast_low = 15,
    nucleus_contrast_high = 99,
    min_non_black_pixels_percentage= 5,
    intensity_threshold=62.0,
    pixel_conv_rate = 0.18,
    diam = 22,
    nucleus_channel_present=True):
    """
    Loads and preprocesses microscopy images for segmentation, including contrast adjustment
    and construction of an RGB image consisting of either one channel or a combo of 2 channels.

    Parameters:
    - main_marker_image_path (str): Path to the image used for segmentation (e.g., red channel).
    - nucleus_image_path (str): Optional path to the image of the nucleus (e.g., DAPI stain).
    - main_marker_channel (str): The fluorescence channel used for segmentation ('red' or 'green').
    - main_marker_contrast_low (int): Lower percentile for contrast stretching of marker channel.
    - main_marker_contrast_high (int): Upper percentile for contrast stretching of marker channel.
    - nucleus_contrast_low (int): Lower percentile for contrast stretching of nucleus channel (if applicable)
    - nucleus_contrast_high (int): Upper percentile for contrast stretching of nucleus channel (if applicable)
    - min_non_black_pixels_percentage (float): Minimum percentage of image area with signal to process.
    - intensity_threshold (float): Intensity threshold above which segmentation channel pixels are considered non-background.
    - pixel_conv_rate (float): Conversion factor from pixel to microns (depends on the microscope that was used to capture images)
    - diam (float): Approximate diameter of the objects in microns (used by Cellpose).
    - nucleus_channel_present (bool): If True, include nucleus image in preprocessing.

    Returns:
    If `nucleus_channel_present` is True:
        - main_marker_image (np.ndarray): Raw segmentation image.
        - nucleus_image (np.ndarray): Raw nucleus image. 
        - diam_pixels (float): Estimated diameter in pixels (used by Cellpose segmentation)
        - marker_channel (int): Index of the RGB channel used for segmentation.
        - rgb_image (np.ndarray): Preprocessed 2-channel color image
    If `nucleus_channel_present` is False:
        - main_marker_image (np.ndarray): Raw segmentation image.
        - diam_pixels (float): Estimated diameter in pixels.
        - marker_channel (int): Index (color) of the RGB channel used for segmentation.
        - rgb_image (np.ndarray): Preprocessed 1-channel color image

    Returns None if the image is considered empty (not enough non-background signal).
    """
    # opening the image file using PIL and converting it to an array
    main_marker_opened = Image.open(main_marker_image_path)
    main_marker_image = np.array(main_marker_opened)
    
    # increasing the contrast of the segmentation channel image
    main_marker_im = increase_contrast_stretch(main_marker_image, main_marker_contrast_low, main_marker_contrast_high)
    
    # counting non-black pixels (see if the minimumm amount of pixels have the intensity above the threshold); skipping the images if the image is empty
    height, width = main_marker_im.shape
    rgb_image = np.zeros((height, width, 3))
    total_image_area = height * width
    non_black_pixel_count = np.sum(main_marker_image > intensity_threshold)
    if non_black_pixel_count < (min_non_black_pixels_percentage / 100) * total_image_area:  # converting the % specified by the user into a pixel number
        return None
    
    # assigning color to the segmentation marker image
    if main_marker_channel == 'red':
        marker_channel = 0
    elif main_marker_channel=='green':
        marker_channel = 1
    rgb_image[..., marker_channel] = main_marker_im 
    diam_pixels = diam / pixel_conv_rate # converting microns specified by the user in pixels

    if nucleus_channel_present:
        nucleus_opened = Image.open(nucleus_image_path)
        nucleus_image = np.array(nucleus_opened) 
        nucleus_im = increase_contrast_stretch(nucleus_image, nucleus_contrast_low, nucleus_contrast_high)
        rgb_image[..., 2] = nucleus_im  # blue channel for the nucleus
        return main_marker_image, nucleus_image, diam_pixels, marker_channel, rgb_image
    else:
        return main_marker_image, diam_pixels, marker_channel, rgb_image


def analyze_segmented_cells(predicted_masks, 
                         main_marker_image, 
                         main_marker_image_name,
                         min_area,
                         pixel_conv_rate, 
                         rgb_image,
                         condition_name,
                         replicate_num,
                         nucleus_image=None,
                         min_nucleus_pixels_percentage=None,  
                         nucleus_pixel_threshold=None,
                         nucleus_channel_present=True, 
                         properties = ['label', 'area', 'bbox_area', 'area_convex', 'perimeter', 'eccentricity', 'extent', 'major_axis_length', 
                        'minor_axis_length', 'centroid', 'mean_intensity', 'max_intensity', 'min_intensity',
                        'equivalent_diameter_area', 'feret_diameter_max' ,'orientation','perimeter_crofton','solidity']):
    """
    Analyze morphological and intensity-based properties of segmented cells 
    and return measurements along with some visualizations.

    Parameters:
    - predicted_masks (np.ndarray): Binary segmentation masks generated by the cellpose.
    - main_marker_image (np.ndarray): Image used for segmentation
    - main_marker_image_name (str): Name of the segmentation image.
    - min_area (float): Minimum area (in µm²) to consider an object a cell
    - pixel_conv_rate (float): Conversion factor from pixels to microns.
    - rgb_image (np.ndarray): RGB image used for generating the overlay.
    - condition_name (str): Experimental condition name.
    - replicate_num (int): Replicate number.
    - nucleus_image (np.ndarray, optional): Image containing nuclear marker.
    - min_nucleus_pixels_percentage (float, optional): Minimum % of nuclear pixels inside the cell that is required to keep a cell
    - nucleus_pixel_threshold (float, optional): Pixel intensity threshold to identify nuclear signal.
    - nucleus_channel_present (bool): If nucleus filtering is to be applied.
    - properties (list): List of region properties to extract.

    Returns:
    - temp_df (pd.DataFrame): DataFrame of valid cell properties.
    - overlay_image (np.ndarray): RGB image with boundaries of selected cells.
    - gray_image (np.ndarray): Grayscale image from the original RGB.
    - mask_list (list): List of dictionaries with binary masks of individual cells.
    """

    labeled_mask = measure.label(predicted_masks)
    cleared_mask = segmentation.clear_border(labeled_mask)
        
    valid_props = []
    valid_regions = []
    mask_list = []  # list that is eventually populated with dictionaries for individual objects
    new_label_counter = 1 # making sure mask labeling starts with 1 after filtering

    for region in measure.regionprops(cleared_mask, intensity_image=main_marker_image):
        region_mask = (cleared_mask == region.label) # extracting a binary mask for the current region
        region_area = region.area 
        valid = region_area > (min_area / pixel_conv_rate**2) # making sure that the cell of the cell is above the minimum

        if nucleus_channel_present: # making sure that the size and the brightness of the nucleus is sufficient for the cell to be considered an object
            min_required_nucleus_pixels = (min_nucleus_pixels_percentage / 100) * region_area
            nucleus_pixels = nucleus_image[region_mask]
            region_nucleus_pixels = np.sum(nucleus_pixels >= nucleus_pixel_threshold)
            valid = valid and (region_nucleus_pixels >= min_required_nucleus_pixels)

        if valid:
            props = {prop: getattr(region, prop) for prop in properties}
            valid_props.append(props)
            valid_regions.append(region)
            mask_list.append({
                "image_name": main_marker_image_name,
                "label": new_label_counter,
                "mask": region_mask
            })
            new_label_counter += 1
    
    # generating grayscale image for further mask overlay
    rgb_image_copy = rgb_image.copy()
    if rgb_image_copy.dtype != np.uint8:
        rgb_image_copy = (rgb_image_copy * 255).astype(np.uint8)
    gray_image = cv2.cvtColor(rgb_image_copy, cv2.COLOR_BGR2GRAY) 
    overlay_image = rgb_image.copy()
    if overlay_image.dtype != np.uint8:
        overlay_image = (overlay_image * 255).astype(np.uint8)
   
    # generating an overlay on top of an image
    for region_data in mask_list:
        mask_data = region_data["mask"]
        image_size = max(rgb_image.shape[:2])  # getting the larger dimension (height or width)
        scaling_factor = image_size / 1000  # adjusting the divisor to control scaling
        thickness = max(1, int(5 * scaling_factor))  # ensuring a minimum thickness of 1
        boundaries = find_boundaries(mask_data, mode='outer')  
        thick_boundaries = dilation(boundaries, disk(thickness))  
        overlay_image[thick_boundaries.astype(bool)] = [255, 255, 255]  

    if valid_props: 
        temp_df = pd.DataFrame(valid_props)
        # overriding skimage-given labels to have them start at 1
        temp_df['label'] = list(range(1, len(temp_df) + 1))
        # cleaning up centroid into two float columns
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
    if array.ndim == 2:  # grayscale image
        channels = 1
    elif array.ndim == 3 and array.shape[2] == 4:  # RGBA image
        channels = 4
    else:
        raise ValueError("Unsupported array shape for conversion to QPixmap.")
    bytes_per_line = width * channels
    q_image = QImage(array.data, width, height, bytes_per_line, format)
    return QPixmap.fromImage(q_image)


def normalize_to_uint8(array):
    """
    Normalize a float or integer array to 8-bit (0–255) for image display.
    """
    array = array - array.min()
    array = (array / array.max() * 255).astype(np.uint8)
    return array


def compute_region_properties(binary_mask, intensity_image=None):
    """
    Compute aggregated morphological and intensity properties for all connected/disconnected and drawn
    masks

    Parameters:
    - binary_mask (np.ndarray): Binary image of one or more objects.
    - intensity_image (np.ndarray, optional): Grayscale image for intensity-based measurements.

    Returns:
    - pd.DataFrame: Single-row DataFrame with aggregate measurements.
    """
    labeled_mask = measure.label(binary_mask.astype(np.uint8))
    props = measure.regionprops(labeled_mask, intensity_image=intensity_image)

    if not props:
        return pd.DataFrame()

    # Total area of all objects combined
    total_area = sum(p.area for p in props)
    if total_area == 0:
        return pd.DataFrame()

    # Helper: weighted average for scalar attrs
    weighted_scalar = lambda attr: sum(p.area * getattr(p, attr) for p in props) / total_area

    # Helper: weighted average for (y, x) centroid
    centroid_y = sum(p.area * p.centroid[0] for p in props) / total_area
    centroid_x = sum(p.area * p.centroid[1] for p in props) / total_area

    result = {
        'label': 1,
        'area': total_area,
        'bbox_area': sum((p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]) for p in props),
        'area_convex': sum(p.convex_area for p in props),
        'perimeter': sum(p.perimeter for p in props),
        'eccentricity': weighted_scalar('eccentricity'),
        'extent': weighted_scalar('extent'),
        'major_axis_length': weighted_scalar('major_axis_length'),
        'minor_axis_length': weighted_scalar('minor_axis_length'),
        'equivalent_diameter_area': weighted_scalar('equivalent_diameter'),
        'feret_diameter_max': max(p.feret_diameter_max for p in props),
        'orientation': weighted_scalar('orientation'),
        'perimeter_crofton': sum(p.perimeter_crofton for p in props),
        'solidity': weighted_scalar('solidity'),
        'centroid_y': centroid_y,
        'centroid_x': centroid_x,
    }
    # Add intensity-related features if an intensity image is provided
    if intensity_image is not None:
        result['mean_intensity'] = np.mean([p.mean_intensity for p in props])
        result['max_intensity'] = max(p.max_intensity for p in props)
        result['min_intensity'] = min(p.min_intensity for p in props)
        
    # Convert the result dictionary to a single-row DataFrame
    return pd.DataFrame([result])


