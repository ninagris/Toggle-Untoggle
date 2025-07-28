![Logo](https://github.com/ninagris/Toggle-Untoggle/blob/main/icons/logo.png)

An Interactive Desktop Application for Cell Segmentation and Single-Cell Morphological Parameter Extraction

"Toggle-Untoggle" workflow:

1. Accepts a single channel (single fluorescence marker or phase-contrast images) or separate pairs of images for the segmentation marker channel and the nucleus channel as input.
2. Segments cells in the images using the cyto3, nuclei, or a custom Cellpose model imported by the user.
3. Extracts morphological parameters and/or ROIs from the selected cells
4. Saves the output to the same folder as the input images. 

## Installation instructions

Note: The package GUI application is currently available only for macOS (M1–M3).

1. Go to the releases page and download the latest version of the software
2. Unzip the downloaded file
3. If your computer prevents the app from opening due to security settings, go to System Settings → Privacy & Security, then allow the app to open under “Apps from unidentified developers”

## Descriptions of the morphological parameters that can be extracted following segmentation

**Note:** All distance and area measurements are reported in microns (µm) or square microns (µm²), based on the pixel-to-micron scale (0-2) provided during segmentation. Parameters are calculated using the `regionprops` function from the [scikit-image](https://scikit-image.org/docs/0.24.x/api/skimage.measure.html#skimage.measure.regionprops) library.

• **Area**: area of the region (i.e. number of pixels of the region scaled by pixel-area).  
• **Area_bbox**: area of the bounding box (i.e. number of pixels of bounding box scaled by pixel-area).  
• **Area_convex**: area of the convex hull image, which is the smallest convex polygon that encloses the region.  
• **Axis_major_length**: the length of the major axis of the ellipse that has the same normalized second central moments as the region.  
• **Axis_minor_length**: the length of the minor axis of the ellipse that has the same normalized second central moments as the region.  
• **Eccentricity**: the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.  
• **Equivalent_diameter_area**: the diameter of a circle with the same area as the region.  
• **Extent**: ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols).  
• **Feret_diameter_max**: maximum Feret’s diameter computed as the longest distance between points around a region’s convex hull contour.  
• **Intensity_max**: value with the greatest intensity in the region.  
• **Intensity_mean**: value with the mean intensity in the region.  
• **Intensity_min**: value with the least intensity in the region.  
• **Orientation**: angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the region, ranging from -π/2 to π/2 counter-clockwise.  
• **Perimeter**: perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.  
• **Perimeter_crofton**: perimeter of object approximated by the Crofton formula in 4 directions.  
• **Solidity**: ratio of pixels in the region to pixels of the convex hull image.


## Post-processing manipulations (optional):

–Haralick texture features extraction:
1. Install Anaconda or Miniconda (https://www.anaconda.com/products/distribution) if you haven't done so yet
2. Download and save Haralick_Feature_Extraction.py file on your computer
3. Open your mac terminal
4. Run the following line: conda create -n haralick_env python=3.11 pandas numpy roifile scikit-image histomicstk -y
5. To activate the environment: conda activate haralick_env
6. Navigate to the location of your script: cd path/to/your/script/folder (replace with your actual folder path, e.g., cd /Users/yourname/Desktop/scripts)
7. Run the following command after replacing the paths with your own: python Haralick_Feature_Extraction.py \                  
  --images_dir "/Users/ninagrishencko/Desktop/test_img/images" \
  --roi_dir "/Users/ninagrishencko/Desktop/test_img/ROIs" \
  --morph_df_path "/Users/ninagrishencko/Desktop/test_img/single_cell_morphology.csv" \
  --save_path "/Users/ninagrishencko/Desktop/test_img/haralick.csv"
8. Once done, you will see the message: "The process is completed!". Large files may take some time to get processed



