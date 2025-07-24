![Logo](https://github.com/ninagris/Toggle-Untoggle/blob/main/icons/logo.png)

An Interactive Desktop Application for Cell Segmentation and Single-Cell Morphological Parameter Extraction

"Toggle-Untoggle" workflow:

1. Accepts separate pairs of images for the segmentation marker channel and the nucleus channel as input
2. Segments cells in the images using the Cellpose Cyto3 generalist model
3. Extracts morphological parameters and/or ROIs from the selected cells
4. Saves the output to the same folder as the input images

## Installation instructions

Note: The application is currently available only for macOS (M1–M3).

1. Go to the releases page and download the latest version of the software
2. Unzip the downloaded file
3. If your computer prevents the app from opening due to security settings, go to System Settings → Privacy & Security, then allow the app to open under “Apps from unidentified developers”

## Post-processing manipulations (optional):

–Haralick texture features extraction:
1. Install Anaconda or Miniconda (https://www.anaconda.com/products/distribution) if you haven't done so yet
2. Download and save Haralick_Feature_Extraction.py file on your computer
3. Open your mac terminal
4. Run the following line: conda create -n haralick_env python=3.11 pandas numpy roifile scikit-image histomicstk -y
5. To activate the environment: conda activate haralick_env
6. Navigate to the location of your script: cd path/to/your/script/folder (replace with your actual folder path, e.g., cd /Users/yourname/Desktop/scripts)
7. Run the following command: Haralick_Feature_Extraction.py
8. The script will prompt you to type in the full paths to your folders and files. Just type the path (no quotes or parentheses) and press Enter.
9. Once done, you will see the message: "The process is completed!". Large files may take some time to get processed



