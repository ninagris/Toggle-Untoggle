![Logo](https://github.com/ninagris/Toggle-Untoggle/blob/main/icons/logo.png)

**An Interactive Desktop Application for Cell Segmentation and Single-Cell Morphological Analysis**

**"Toggle-Untoggle" workflow:**
1. Accepts a single channel (single fluorescence marker or phase-contrast images) or separate pairs of images for the segmentation marker channel and the nucleus channel as input.
2. Segments cells in the images using the cyto3, nuclei, or a custom Cellpose model imported by the user.
3. Extracts morphological parameters and/or ROIs from the selected cells
4. Saves the output to the same folder as the input images. 

## Installation Instructions

**Note:** The packaged GUI application can currently be run without external hardware only on macOS (M1–M3), as it relies on embedded GPU resources available on these devices. 

1. Go to the releases page and download the latest version of the software
2. Unzip the downloaded file
3. If your computer prevents the app from opening due to security settings, go to System Settings → Privacy & Security, then allow the app to open under “Apps from unidentified developers”

## Setting Up an Anaconda Environment

**Note:** If you encounter any problems with version incompatibilities or missing packages, try modifying the environment.yml file or install missing packages using pip.

1. **Install Anaconda or Miniconda**  
   If you haven’t installed it yet, download and install from  
   [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution).
2. **Clone the repository**  
   • Go to the repository page in your browser: https://github.com/ninagris/Toggle-Untoggle  
   • Click the green **Code** button, then select **Download ZIP**  
   • Once downloaded, unzip the file
3. **Navigate into the project folder**  
   Open your terminal or command prompt and navigate to the directory where you saved the unzipped file:  
   ```bash
   cd path/to/Toggle-Untoggle
4. **Create the conda environment from the YAML file**  
   Open your terminal or command prompt and run:  
   ```bash
   conda env create -f environment.yml
5. **Activate the conda environment**  
   Open your terminal or command prompt and run:  
   ```bash
   conda activate toggle-untoggle
6. **Launch the application**  
   Open your terminal or command prompt and run:  
   ```bash
   python main.py

## GUI Input Parameters:

1. **Images Folder Path:**
The path to the folder containing the images. Only single-channel raw images should be included.
If multi-channel images with the same file IDs are present, you may see a “No images have been processed” message.
Files without the .tif, .tiff, .TIF, or .TIFF extensions will be ignored.
**⚠️ Important**:
–Ensure the folder does not contain both raw and display images; only raw images should be used as input.
–Perform maximum projection or other image preprocessing manipulations before using Toggle-Untoggle.
–If processing both segmentation and nucleus channels, make sure the images appear in matching pairs within the folder. For example:
      •ntc 0.3 auto_Top Slide_R_p00_0_A01f04d2.TIF
      •ntc 0.3 auto_Top Slide_R_p00_0_A01f04d0.TIF
      •ntc 0.3 auto_Top Slide_R_p00_0_A01f38d2.TIF
      •ntc 0.3 auto_Top Slide_R_p00_0_A01f38d0.TIF
where the suffixes like d2 and d0 indicate different channels. 
2. **Output File Name:**
The desired name for the output .csv file. This is not a file path, just the name without the .csv extension.
The file will be saved in the Images Folder. The output fle name can be changed to allow for multiple savings.
3.**ROI Folder Name:**
The desired name for the folder that will contain the ROIs of the selected cells. This is not a folder path, just the name.
The folder will be created inside the Images Folder. The ROI folder name can be changed to allow for multiple savings.
4. **Cellpose Model:**
Select one of the available options or enter the path to your custom pre-trained model saved locally. Refer to [Cellpose guidelines](https://rdcu.be/eypEB) on how to train your own model.
5. **Condition Name:**
Additional column with the specified condition/treatment/experiment name that will be added to the .csv file. 
6. **Replicate #:**
Additional column with the replicate number specified that will be added to the .csv file.
7. **Segmentation Channel File ID:**
A keyword unique to images containing a segmentation marker (e.g., d2, ch1). 
8. **Display Cell Labels:**
If checked, labels will be displayed on top of each segmented object using the specified font for the digits.
9. **Nucleus Channel Present:**
If checked, additional input fields for specifying the nucleus channel input parameters will be displayed including:
   **–Nucleus Channel File ID:** A keyword unique to images containing a nuclear marker (e.g., d0, ch2).
   **–Lower Percentile of Pixel Intensities for Nucleus Channel:** Any intensity below this percentile is mapped to 0 (black). Contrast adjustments are for visualization only; fluorescence intensity is extracted from the original images you input.
   **–Upper Percentile of Pixel Intensities for Nucleus Channel:** Any intensity above this percentile is mapped to 1 (white). 
   **–Minimum Percentage of Cell Area Occupied by Nucleus:** The minimum proportion of the cell's area that must be occupied by nucleus (blue) pixels.
   **–Nucleus Channel Intensity Threshold:** Minimum fluorescence intensity for a pixel to be considered part of the nucleus.
10. **Segmentation Channel Color:**
The color of the segmentation channel for display (choose from dropdown). The color is only used for the isualization purposes.
11. **Lower Percentile of Pixel Intensities for Segmentation Marker Channel:**
Any intensity below this percentile is mapped to 0 (black). Contrast adjustments are for visualization only; fluorescence intensity is extracted from the original images that you input.
12. **Upper Percentile of Pixel Intensities for Segmentation Marker Channel:**
Any intensity above this percentile is mapped to 1 (white).
13. **Average Cell Diameter:**
The typical cell diameter in microns. 
14. **Flow Threshold:**
Maximum allowed flow error per segmented mask. 
15. **Minimum Cell Area:**
Minimum area (in microns²) for a segmented object to be considered a valid cell.
16. **Minimum Percentage of Image Occupied by Cells:**
Minimum proportion of the image that must be covered by cells. 
17. **Segmentation Channel Intensity Threshold:**
Minimum fluorescence intensity required for a pixel to be included in segmentation. 
18. **Pixel-to-Micron Ratio:**
The conversion factor from pixels to microns (depends on your microscope setup). The allowed input range is [0,2]

**Toggle Mode:** Go through your images and click on the cells you want to exclude from the analysis. Clicking again will re-include the cells, even if you’ve already pressed the save button. Untoggled cells appear dimmer than active ones included in the analysis.

**Connect Mode:** Go through your images and draw a line with your mouse across two or more cells you want to merge. Once the masks share the same color, the cells are considered connected. To disconnect them, draw a line between the same cells again.

**Draw Mode and Erase Mode:** Go through your images and use the pen tool to draw missing cells or the eraser tool to remove cells you've just drawn. Make sure to draw closed shapes to ensure accurate measurements.

## Combining Multiple .csv files into one
1. Ensure that python is installed on your machine. You can check this by running python --version or python3 --version in your terminal.
2. Open a terminal or command prompt.
3. Navigate to the folder containing your .csv files and combined_csvs.py (part of the repo) using the cd command. Make sure this folder only contains the .csv files you wish to combine.
   ```bash
   cd path/to/my_csv_folder
4. Install pandas if it isn’t already installed:
   ```bash
   pip install pandas
5. Run the python script (you may need to use python3 instead):
   ```bash
   python combined_csvs.py
6. The combined file named combined.csv will be created in the same folder.

## Descriptions Of the Morphological Parameters Extracted Following Segmentation

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

## Troubleshooting Commonly Encountered Issues

## Troubleshooting Commonly Encountered Issues

| **Issue**                                                                                      | **Suggested Solution**                                                                                         |
|------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Empty images are displayed                                                                     | Increase the segmentation channel intensity threshold and/or the minimum percentage of the image occupied by cells. |
| Sluggish performance or frequent freezing                                                      | Use additional GPU resources if available.                                                                    |
| Tiny objects that are not cells are segmented                                                  | Increase the minimum cell area.                                                                               |
| Too few cells are detected                                                                     | Increase the `flow_threshold` and/or adjust the average cell diameter.                                        |
| Many incorrectly detected cells                                                                | Decrease the `flow_threshold` and/or adjust the average cell diameter.                                        |
| Freezing after processing a certain percentage of images on macOS native GPU                  | Split images into separate folders and process in smaller batches, or use additional GPU resources.           |
| Drawn ROI measurements appear unrealistically large or small                                   | Ensure ROIs are drawn as enclosed objects as accurately as possible.                                          |
| Display color images are too dim or over-saturated                                             | Adjust the maximum and/or minimum percentile for the corresponding channel using the sliders.                |



## Toggle-Untoggle Limitations ⚠️🛠:

- **Subjectivity:** Manual corrections may introduce user bias.  
- **Low throughput:** Not well-suited for large-scale screening.  
- **Hardware-dependent:** Performance is limited when using only a CPU.  
- **Limited cross-platform testing:** Primarily tested on macOS; performance on other operating systems is uncertain.

## Post-Processing Manipulations (optional):

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



