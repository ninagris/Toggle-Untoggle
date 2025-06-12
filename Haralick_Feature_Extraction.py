import pandas as pd
import numpy as np
import re
import os
import zipfile
import tempfile
import warnings
warnings.filterwarnings('ignore')

from io import BytesIO
from roifile import ImagejRoi
from skimage.draw import polygon
from skimage.measure import regionprops
from skimage import io
from histomicstk.features import compute_haralick_features 


# **Defining Paths**
# ```
# /Users/ninagrishencko/Desktop/A2780vsA2780CisR/
# └── A2780/
#     ├── images/
#     │   ├── r02c02f03_MAX_ch2.tiff
#     │   ├── r02c02f03_MAX_ch3.tiff
#     │   └── ...
#     ├── ROIS/
#     │   ├── r02c02f03_MAX_ch2.zip
#     │   ├── r02c02f02_MAX_ch2.zip
#     │   └── ...
#     └── single_cell_morphology.csv
# ```

images_dir = input("Please enter the path to your images folder: ")
roi_dir = input("Please enter the path to your ROI folder: ")
morph_df_path = input("Please enter the path to your morphology CSV file: ")
save_path = input("Please enter the path where you want to save the output CSV: ")

morph_df = pd.read_csv(morph_df_path, index_col=False)
morph_df['image_name'] = morph_df['image_name'].str.replace(r"\.tiff?$", "", flags=re.IGNORECASE, regex=True)


def extract_rois_from_zip(roi_dir, image_name):
    """
    Extracts ROI files from a ZIP folder with ROIs into memory.

    Returns:
        dict: Keys are ROI filenames (with .roi), values are BytesIO binary objects.
    """
    zip_path = os.path.join(roi_dir, f'{image_name}.zip') 
    rois = {}
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist(): # a list of all filenames inside the ZIP
            if file.endswith('.roi'): 
                with zip_ref.open(file) as roi_file:
                    rois[file] = BytesIO(roi_file.read()) # Wrapping binary data into an in-memory file-like object
    return rois


def convert_roi_bytes_to_mask(roi_bytes, im_shape):
    """
    Convert an ImageJ ROI stored as in-memory bytes into a 2D binary mask.

    Parameters:
        roi_bytes (BytesIO): In-memory binary file containing the ROI file data.
        im_shape (tuple): Shape of the target image (height, width) to create the mask.

    Returns:
        np.ndarray: A 2D boolean NumPy array where pixels inside the ROI are True, others False.

    """

    with tempfile.NamedTemporaryFile(suffix='.roi') as tmp: # Creating a temporary file with suffix .roi to write the ROI bytes into disk
        tmp.write(roi_bytes.getbuffer())  # Writing the bytes from the in-memory ROI to the temporary file
        tmp.flush() 
        roi = ImagejRoi.fromfile(tmp.name) # Loading the ROI from the temporary file using roifile's ImagejRoi

    # Getting the coordinates of the ROI's polygon points 
    coords = roi.coordinates()
    x = coords[:, 0]
    y = coords[:, 1]

    # Filling out the area inside of ROI to generate a binary mask
    mask = np.zeros(im_shape, dtype=bool) 
    r, c = polygon(y, x, shape=im_shape)
    mask[r, c] = True

    return mask


def compute_haralick(full_im, cell_mask, 
                                 # Offsets are list of pixel displacements to compute co-occurrence matrices at multiple scales and directions.
                                 offsets=np.array([ [0, 1], [1, 0], [1, 1], [-1, 1],   # Small scale, 1 pixel apart
                                                   [0, 5], [5, 0], [5, 5], [-5, 5]]),   # Medium scale
                                 num_levels=64, # Number of gray levels to quantize the image intensities to
                                 clip_percentiles=(1, 99)): #  low and high percentiles used to clip intensity values to avoid outliers affecting quantization.

    """
    Computing haralick texture features for a single object after pre-processing

    """

    # Converting a boolean mask into a binary mask
    int_mask = np.zeros_like(cell_mask, dtype=np.uint8)
    int_mask[cell_mask] = 1

    int_vals = full_im[cell_mask] # Getting intensity values inside ROI
    low, high = np.percentile(int_vals, clip_percentiles) # clipping range for the ROI

    im_clipped = np.clip(full_im, low, high) # Limiting pixel inside the ROI to the clipping range
    im_quant = ((im_clipped - low) / (high - low) * (num_levels - 1)).astype(np.uint8)  # Quantizing image into [0, num_levels - 1]

    # Computing Haralick texture features
    df = compute_haralick_features(
        int_mask,
        im_quant,
        offsets=offsets,
        num_levels=num_levels, 
        gray_limits=[0, num_levels - 1], # Gray limits of the quantized images
    )

    return df.iloc[0].to_dict()


def match_rois_with_df_objects(df_subset, haralick_feats):
    """
    Matches ROI-based Haralick features to segmented objects in df_subset using 'roi_name' as key.

    Parameters:
        df_subset (pd.DataFrame): Subset of morph_df for the current image with segmented objects.
                                  Must contain a 'roi_name' column.
        haralick_feats (list of dict): List of Haralick feature dictionaries, each with a 'roi_name' key.

    Returns:
        pd.DataFrame: A new DataFrame with Haralick features appended, matched by 'roi_name'.

    """
    haralick_df = pd.DataFrame(haralick_feats)

    # Ensuring both haralick_df and df_subset have 'roi_name' column and merge on it
    if 'roi_name' not in df_subset.columns or 'roi_name' not in haralick_df.columns:
        raise ValueError("'roi_name' column is required in both df_subset and haralick_feats.")
    matched_df = df_subset.merge(haralick_df, on='roi_name', how='left')

    return matched_df


def move_cols(df):
    """
    Reorders specific identifier columns in the DataFrame to improve readability and consistency.

    This function moves the 'Condition', 'Replicate', and 'roi_name' columns directly after the 'label' column. 
    These columns may have been appended to the end or scattered in the DataFrame due to earlier processing steps 
    (e.g., merging or appending new features).
    """
    cols = list(df.columns)
    cols_to_move = ["roi_name", "Condition", "Replicate", ]

    # Remove them from the current list
    for col in cols_to_move:
        cols.remove(col)

    # Find index of 'label'
    label_idx = cols.index("label")

    # Insert the cols_to_move right after the 'label column'
    for i, col in enumerate(cols_to_move):
        cols.insert(label_idx + 1 + i, col)

    # Reorder the DataFrame
    final_df = df[cols]

    return final_df

haralick_dfs = []

for image_name in morph_df['image_name'].unique():

    # Reading the image
    image_path = os.path.join(images_dir, f'{image_name}.tif')
    im_intensity = io.imread(image_path, plugin='tifffile', key=0)

    # Extracting ROIs for this image
    rois = extract_rois_from_zip(roi_dir, image_name)

    # Subset morph_df for this image
    df_subset = morph_df[morph_df['image_name'] == image_name].copy()

    # Ensure roi_name column is in df_subset
    if 'roi_name' not in df_subset.columns:
        df_subset['roi_name'] = df_subset.apply(
            lambda row: f"{os.path.splitext(row['image_name'])[0]}_label{row['label']}.roi", axis=1
        )

    haralick_feats = []

    for fname, roi_bytes in rois.items():
        roi_mask = convert_roi_bytes_to_mask(roi_bytes, im_intensity.shape)
        feats = compute_haralick(im_intensity, roi_mask)
        feats['roi_name'] = fname

        haralick_feats.append(feats)

    matched_df = match_rois_with_df_objects(df_subset, haralick_feats)
    haralick_dfs.append(matched_df)

# Combine all matched dataframes
concat_df = pd.concat(haralick_dfs, ignore_index=True)
haralick_df = move_cols(concat_df)

# Save final dataframe to the specified directory
haralick_df.to_csv(save_path, index=False)
print("The process is completed!")





