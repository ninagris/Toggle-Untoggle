import sys
import numpy as np
import pandas as pd
import os
import torch
import roifile
import zipfile
import cv2
import os
import shutil
import platform

from skimage import measure
from cellpose import models
from PIL import Image
from tqdm import tqdm

from PyQt6.QtWidgets import QApplication, QLabel, QPushButton
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QMainWindow
from PyQt6.QtWidgets import QTabWidget, QScrollArea, QSizePolicy, QStyleFactory, QCheckBox
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer,QSize, qInstallMessageHandler

from image_analysis_pipeline import open_folder, image_preprocessing, analyze_segmented_cells, convert_to_pixmap, normalize_to_uint8, pixel_conversion, compute_region_properties
from toggle import ImageViewer, ViewerModeController
from input_form_components import InputFormWidget, DraggableTextEdit

def handler(mode, context, message):
    if "Could not parse stylesheet" in message or "QLayout::addChildLayout" in message:
        return  # suppress

qInstallMessageHandler(handler)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller bundle """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def resource_path_2(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Load default Cellpose model (cyto3) with GPU enabled
        self.model = models.CellposeModel(gpu=True, model_type='cyto3')

        # Window setup
        self.setWindowTitle("Toggle-Untoggle")
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowIcon(QIcon(resource_path_2("icons/icon.ico")))
        self.resize(850, 700)
        self.setMinimumSize(850, 700)
        #self.showFullScreen()

        # Controller for managing viewer mode states
        self.viewer_mode_controller = ViewerModeController()
        # Main tab layout
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.setCentralWidget(self.tabs)
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                font-family: Arial;
                font-size: 16pt;  /* Adjust tab font size */
                padding: 2px;    /* Increase padding for bigger tabs */
                min-width: 200px; /* Set a minimum width for tabs */
                min-height: 30px; /* Set a minimum height for tabs */
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: lightblue;  /* Highlight selected tab */
            }
        """)

        # GitHub link in top-right corner
        self.github_link = QLabel()
        self.github_link.setText(
            '<a href="https://github.com/ninagris/Toggle-Untoggle" style="font-size:18px;">'
            'Our GitHub</a>'
        )
        self.github_link.setTextFormat(Qt.TextFormat.RichText)
        self.github_link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.github_link.setOpenExternalLinks(True)
        self.github_link.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.github_link.setStyleSheet("padding-right: 15px;")
        self.tabs.setCornerWidget(self.github_link, Qt.Corner.TopRightCorner) # Adding the link to the top right corner of the tab bar

        # Help text overlay
        self.help_text = DraggableTextEdit(parent=self)
        self.help_text.setVisible(False)
        font = QFont("Arial", 18) 
        self.help_text.setFont(font)
        self.help_text.setReadOnly(True)

        # Input tab and form
        self.input_tab = QWidget() 
        self.input_tab.setLayout(QVBoxLayout())
        self.input_form = InputFormWidget(parent=self.input_tab, help_text=self.help_text, tab_widget=self.tabs)
        self.input_tab.layout().addWidget(self.input_form)
        self.tabs.addTab(self.input_tab, "Input Parameters")

        # State variables
        self.gray_viewers = []
        self.processing_in_progress = False
        self.input_form.processClicked.connect(self.on_process_clicked)
        self.input_form.saveClicked.connect(self.collect_all_callbacks)
        self.images_tab = None   # Image display tab
        self.image_layout = None  # Layout for processed image widgets
    

    def update_modes(self):
        """
        Handle mutually exclusive mode selection (toggle, connect, draw, erase)
        """
        mode = None  # Initialize mode
        sender = self.sender()
        # Toggle mode
        if sender == self.toggle_checkbox and sender.isChecked():
            self.correction_checkbox.setChecked(False)
            self.drawing_checkbox.setChecked(False)
            self.erase_checkbox.setChecked(False)
            mode = "toggle"
        # Connect (correction) mode
        elif sender == self.correction_checkbox and sender.isChecked():
            self.toggle_checkbox.setChecked(False)
            self.drawing_checkbox.setChecked(False)
            self.erase_checkbox.setChecked(False)
            mode = "connect"
        # Draw new masks mode
        elif sender == self.drawing_checkbox and sender.isChecked():
            self.toggle_checkbox.setChecked(False)
            self.correction_checkbox.setChecked(False)
            self.erase_checkbox.setChecked(False)
            mode = 'draw'
        # Erase drawn masks mode
        elif sender == self.erase_checkbox and sender.isChecked():
            self.toggle_checkbox.setChecked(False)
            self.correction_checkbox.setChecked(False)
            self.drawing_checkbox.setChecked(False)
            mode = 'erase'

        self.viewer_mode_controller.set_mode(mode)  

    def is_gpu_available(self):
        if torch.cuda.is_available():
            return True
        elif torch.backends.mps.is_available():  # macOS Metal support
            return True
        return False

    def handle_model_selection(self, model_type: str, custom_path: str):
        """
        Loads a Cellpose model based on dropdown or custom path.
        """
        self.model = None # Clear previous model
        try:
            # Get user GPU preference from checkbox
            user_wants_gpu = self.input_form.GPU_checkbox.isChecked()
            gpu_available = self.is_gpu_available()
            use_gpu = user_wants_gpu and gpu_available

            if user_wants_gpu and not gpu_available:
                self.update_status_label("You selected GPU usage, but no compatible GPU is available. Falling back to CPU.")

            # Load model
            if custom_path:
                self.model = models.CellposeModel(gpu=use_gpu and torch.cuda.is_available(),
                                                pretrained_model=custom_path)
            elif model_type in ["cyto3", "nuclei", "livecell_cp3"]:
                self.model = models.CellposeModel(gpu=use_gpu and torch.cuda.is_available(),
                                                model_type=model_type)
            else:
                raise ValueError("Invalid model type selection.")

            # Manually set to MPS if requested and available
            if use_gpu and platform.system() == 'Darwin' and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.model.device = torch.device("mps")


        except Exception as e:
            self.model = None
            #print(f"[MODEL] Model loading failed: {e}")
            raise


    def resizeEvent(self, event):
        """
        Maintain 20:13 aspect ratio for fullscreen display
        """
        current_size = self.size()
        aspect_ratio = 20 / 13
        desired_width = current_size.width()
        desired_height = int(desired_width / aspect_ratio)

        if desired_height > current_size.height():
            desired_height = current_size.height()
            desired_width = int(desired_height * aspect_ratio)

        if (desired_width, desired_height) != (current_size.width(), current_size.height()):
            self.resize(desired_width, desired_height)
            return  # Skip recentering during forced resize

        super().resizeEvent(event)
        # Recenter help window if visible
        if hasattr(self, "help_text") and self.help_text.isVisible():
            # Defer centering so all layouts & sizes update first
            QTimer.singleShot(0, self.help_text.center_in_parent)


    def on_process_stop(self):
        """
        Stop processing by killing the worker
        """
        if self.worker is not None:
            self.worker.stop()  # Setting the abort flag to stop the worker
            self.input_form.processing_label.setText("Stopping...")

        if self.stop_button is not None:
            self.stop_button.setDisabled(True)  # Preventing accidental/multiple clicks

        if self.worker is None:
            # Enabling the process button again
            self.input_form.process_button.setEnabled(False)
            self.processing_in_progress = False


    def update_stop_button(self, count, button_layout):
        """
        Show stop button only once, after first image processed
        """
        if count > 0 and (not hasattr(self, "stop_button") or self.stop_button is None):  # Making sure that the stop button is only created once
            self.stop_button = QPushButton("Stop Processing")
            self.stop_button.setStyleSheet("""
                QPushButton {
                    font-size: 20pt;  /* Bigger font size */
                    padding: 5px;    /* Add padding around the text */
                    background-color: #FFD1DC; /* Pale pink background */
                    color: black;     /* Black text */
                    border-radius: 5px; /* Rounded corners */
                    border: 1px solid #ddd; /* Border around button */
                }
                QPushButton:hover {
                    background-color: #FBC8D4;  /* Slightly lighter pink on hover */
                }
                QPushButton:pressed {
                    background-color: #FFB0C3;  /* Slightly darker pink when pressed */
                }
                """)
            self.stop_button.setFont(QFont("Arial", 25))
            self.stop_button.setMinimumSize(20, 20)
            self.stop_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
            button_layout.addWidget(self.stop_button)
            button_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.stop_button.clicked.connect(self.on_process_stop)
            self.input_form.process_button.setEnabled(False)


    def on_process_clicked(self, button_layout):
        """
        Start processing when user clicks 'Process'
        """
        if self.processing_in_progress:
            return
        # Reset image viewer data
        if hasattr(self, "gray_viewers"):
            for viewer in self.gray_viewers:
                viewer.connected_groups.clear()
                if hasattr(viewer, "mask_id_to_group"):
                    viewer.mask_id_to_group.clear()
                if hasattr(viewer, "new_mask_dict"):
                    viewer.new_mask_dict.clear()
            self.gray_viewers.clear()  # Remove old viewers
        # Load model from UI
        try:
            # Get the current values live from the widget
            current_model_text = self.input_form.model_selector_widget.model_dropdown.currentText()

            if current_model_text == "custom model":
                model_type = ""
                custom_path = self.input_form.model_selector_widget.custom_model_input.text().strip()
                if not custom_path: # checking if the input is completely empty
                    self.update_status_label("Please input a path to the custom model!")
                    self.processing_in_progress = False
                    self.input_form.process_button.setEnabled(True)
                    return
            else:
                model_type = current_model_text
                custom_path = ""

            self.handle_model_selection(model_type, custom_path)

            if self.model is None:
                raise ValueError("No valid model loaded. Check your selection.")

        except Exception as e:
            self.update_status_label(f"Model loading error: {str(e)}")
            self.processing_in_progress = False
            self.input_form.process_button.setEnabled(True)
            return
        
        # Proceed with processing
        self.processing_in_progress = True
        self.input_form.processing_label.setText("Processing started...")
        self.input_form.process_button.setEnabled(False)
        
        # Clearing the existing images tab before starting the processing
        self.stop_button = None 
        if self.images_tab is not None: 
            current_index = self.tabs.indexOf(self.images_tab)
            if current_index != -1:
                self.tabs.removeTab(current_index)
                self.images_tab = None
                self.image_layout = None

        # Starting processing after clearing the old tab
        self.start_processing()
        if self.worker is not None: # Stop buttons appears only if at least one image has been processed
            self.worker.image_processed.connect(lambda *args: 
                self.update_stop_button(self.worker.count, button_layout) if hasattr(self, "stop_button") and self.stop_button is None else None
            )
            self.worker.finished.connect(lambda: self.stop_button.deleteLater() if hasattr(self, "stop_button") and self.stop_button else None)
    
    def create_images_tab(self):
        """
        Create a new tab for images
        """
        images_tab = QWidget()
        self.tabs.addTab(images_tab, "Processed Images")  # Add the new tab to the widget
        layout = QVBoxLayout()
        # Scrollable area for images
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        image_container = QWidget()
        image_layout = QVBoxLayout()
        image_container.setLayout(image_layout)
        scroll_area.setWidget(image_container)
        image_container.setStyleSheet("background-color: white;")  
       
        self.single_cell_checkbox = QCheckBox("Single-cell morphology")
        self.roi_checkbox = QCheckBox("ROIs")
        self.single_cell_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 18pt;  /* Adjust font size for the label */
                padding: 5px;  /* Optional, adjust for better spacing */
            }
            QCheckBox::indicator {
                width: 25px;  /* Adjust size of the checkbox */
                height: 25px;
            }
        """)
        self.roi_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 18pt;  /* Adjust font size for the label */
                padding: 5px;  /* Optional, adjust for better spacing */
            }
            QCheckBox::indicator {
                width: 25px;  /* Adjust size of the checkbox */
                height: 25px;
            }
        """)
        # Creating save button
        self.save_button = QPushButton("Save")
        self.save_button.setMinimumSize(100, 40) 
        self.save_button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.save_button.setStyleSheet("""
            QPushButton {
                font-size: 20pt;  /* Bigger font size */
                padding: 10px;    /* Add padding around the text */
                background-color: #ADD8E6; /* Light blue background */
                color: black;     /* Black text */
                border-radius: 5px; /* Rounded corners */
                border: 1px solid #ddd; /* Border around button */
            }
            QPushButton:hover {
                background-color: #87CEEB;  /* Slightly darker blue on hover */
            }
            QPushButton:pressed {
                background-color: #4682B4;  /* Even darker blue when pressed */
            }
        """)
        # Creating a QWidget to hold the button and label
        self.button_widget = QWidget()
        button_layout = QVBoxLayout(self.button_widget)
        button_layout.addWidget(self.single_cell_checkbox)
        button_layout.addWidget(self.roi_checkbox)
        button_layout.addWidget(self.save_button)
        button_layout.setSpacing(5)
        button_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        self.save_button.clicked.connect(self.collect_all_callbacks)
        checkbox_row = QHBoxLayout()
        checkbox_row.setAlignment(Qt.AlignmentFlag.AlignRight)  # Align content inside the row to the right

        self.toggle_checkbox = QCheckBox("Toggle")
        self.correction_checkbox = QCheckBox("Connect")
        self.drawing_checkbox = QCheckBox()
        self.drawing_checkbox.setIcon(QIcon(resource_path("icons/pen.png")))
        self.drawing_checkbox.setIconSize(QSize(25, 25))  # adjust icon size
        self.drawing_checkbox.setText("")
        self.erase_checkbox = QCheckBox()
        self.erase_checkbox.setIcon(QIcon(resource_path("icons/eraser.png")))
        self.erase_checkbox .setIconSize(QSize(30, 30))  # adjust icon size
        self.erase_checkbox .setText("")
        self.toggle_checkbox.setChecked(True)

        # Styling (optional)
        self.toggle_checkbox.setStyleSheet("""
            QCheckBox { font-size: 18pt; padding: 5px; }
            QCheckBox::indicator { width: 25px; height: 25px; }
        """)
        self.correction_checkbox.setStyleSheet("""
            QCheckBox { font-size: 18pt; padding: 5px; }
            QCheckBox::indicator { width: 25px; height: 25px; }
        """)
        self.drawing_checkbox.setStyleSheet("""
            QCheckBox { font-size: 18pt; padding: 5px; }
            QCheckBox::indicator { width: 25px; height: 25px; }
        """)
        self.erase_checkbox.setStyleSheet("""
            QCheckBox { font-size: 18pt; padding: 5px; }
            QCheckBox::indicator { width: 25px; height: 25px; }
        """)

        self.toggle_checkbox.stateChanged.connect(self.update_modes)
        self.correction_checkbox.stateChanged.connect(self.update_modes)
        self.drawing_checkbox.stateChanged.connect(self.update_modes)
        self.erase_checkbox.stateChanged.connect(self.update_modes)
        checkbox_row.addWidget(self.toggle_checkbox)
        checkbox_row.addWidget(self.correction_checkbox)
        checkbox_row.addWidget(self.drawing_checkbox)
        checkbox_row.addWidget(self.erase_checkbox)

        checkbox_container = QWidget()
        checkbox_container.setLayout(checkbox_row)
        checkbox_container.setFixedHeight(60)
        layout.addWidget(checkbox_container, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(checkbox_row)
        layout.addWidget(scroll_area)
        layout.addWidget(self.input_form.progress_bar)  # Adding progress bar below the scroll area
        images_tab.setLayout(layout)
        # Ensuring the layout is cleared before returning it
        self.clear_layout(image_layout)
        self.images_tab = images_tab  # Updating the reference to the new tab
        self.image_layout = image_layout 

        return image_layout 
    

    def resource_path(self, relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller bundle """
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)
    
    def clear_layout(self,layout):
        """
        Helper function to clear all widgets in a layout
        """
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()


    def collect_all_callbacks(self):
        """
        Main method to process mask callbacks and export data.
        Depending on the user selections, this function filters out inactive masks,
        saves the active properties to a CSV file, and/OR generates ROI ZIP files.
        """
        save_rois = self.roi_checkbox.isChecked()
        save_csv = self.single_cell_checkbox.isChecked()
        correct_props_df, excluded_props_df = self.filter_active_props()
        if save_csv:
            self.save_props_to_csv(correct_props_df, excluded_props_df)
        if save_rois:
            self.save_rois_to_zip(correct_props_df)


    def filter_active_props(self):
        """
        Filters single-cell properties to separate active and inactive object entries.

        Returns:
            correct_df (pd.DataFrame): DataFrame containing active object properties.
            excluded (pd.DataFrame): DataFrame of inactive object properties.
        """
        all_props_df = self.worker.all_props_df.copy()
        all_props_df['combined_key'] = all_props_df['image_name'].astype(str) + '_' + all_props_df['label'].astype(str)

        active_keys = set()
        inactive_keys = set()
        # Collect active/inactive states from all viewers
        for viewer in self.gray_viewers:
            for cb_key, cb_data in viewer.callback_dict.items():
                if not cb_data.get("is_active"):
                    inactive_keys.add(cb_key)
                    continue
                active_keys.add(cb_key)

        correct_df = correct_df = all_props_df[all_props_df.apply(
            lambda r: f"{r['image_name']}_{r['label']}" in active_keys, axis=1
        )]
        excluded = all_props_df[all_props_df.apply(
            lambda r: f"{r['image_name']}_{r['label']}" in inactive_keys, axis=1
        )]

        return correct_df, excluded

    
    def save_props_to_csv(self, correct_df, excluded_df):
        """
        Saves filtered single-cell properties to CSV files.
        
        Parameters:
            correct_df (pd.DataFrame): DataFrame of active, valid objects to keep.
            excluded_df (pd.DataFrame): DataFrame of filtered-out (inactive) objects (placeholder, will be overwritten).
        """  
       # Collect active/inactive keys
        active_keys = set()
        inactive_keys = set()

        for viewer in self.gray_viewers:
            for cb_key, cb_data in viewer.callback_dict.items():
                if not cb_data.get("is_active"):
                    inactive_keys.add(cb_key)
                    continue
                active_keys.add(cb_key)

        # Integrate newly drawn, merged, or disconnected masks
        combined_df, excluded_df = self.integrate_new_objects(correct_df, active_keys, inactive_keys)
        if self.roi_checkbox.isChecked():
            combined_df = self.add_roi_name_column(combined_df)
            excluded_df = self.add_roi_name_column(excluded_df)

        # Save final CSV files
        output_dir =self.input_form.images_folder_path.text()
        output_csv = os.path.join(output_dir, self.input_form.csv_file_name.text() + '.csv')
        combined_df.to_csv(output_csv, index=False)

        excluded_csv = os.path.join(output_dir, 'excluded_objects.csv')
        if not excluded_df.empty:
            excluded_df.to_csv(excluded_csv, index=False)
        else:
            if os.path.exists(excluded_csv):
                os.remove(excluded_csv)

    def gather_active_keys(self):
        """
        Returns a set of active keys across all viewers.
        """
        active_keys = set()
        for viewer in self.gray_viewers:
            for cb_key, cb_data in viewer.callback_dict.items():
                if cb_data.get("is_active", False):
                    active_keys.add(cb_key)
        return active_keys
    
    def prepare_roi_dir(self, folder_path, folder_name):
        """
        Prepares the directory for saving ROI files by clearing existing contents.

        Parameters:
            folder_path (str): Parent directory path.
            folder_name (str): Subfolder name for ROIs.

        Returns:
            str: Full path to the created ROI directory.
        """
        roi_dir = os.path.join(folder_path, folder_name)
        if os.path.exists(roi_dir):
            shutil.rmtree(roi_dir)
        # create a fresh directory
        os.makedirs(roi_dir, exist_ok=True)
        return roi_dir
    
    def filter_active_labels(self, df, active_keys):
        """
        Filters a DataFrame to keep only rows with valid, non-empty labels present in active_keys.

        Parameters:
            df (pd.DataFrame): Input properties DataFrame.
            active_keys (set): Set of active image_label keys.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        return df[
            df.apply(
                lambda r: f"{r['image_name']}_{r['label']}" in active_keys
                and pd.notnull(r['label'])
                and str(r['label']).strip() != '',
                axis=1
            )
        ]
    
    def collect_masks(self, df):
        """
        Reconstructs mask arrays for active objects to prepare for ROI export.

        Parameters:
            df (pd.DataFrame): Filtered properties DataFrame.

        Returns:
            image_masks_dict (dict): Dict of image_name → labeled mask array.
            label_map (dict): Dict of combined_key → label value.
        """
        image_masks_dict = {}
        label_map = {}

        for _, row in df.iterrows():
            image_name = row['image_name']
            label_str = str(row['label'])
            mask_key = f"{image_name}_{label_str}"
            # Retrieve mask from viewer or worker
            mask = self.get_mask_by_key(mask_key)
            if mask is None or np.sum(mask) == 0:
                continue

            label_value = self.assign_label_value(label_str, label_map)
            label_map[mask_key] = label_value
            # Initialize base mask array
            if image_name not in image_masks_dict:
                image_masks_dict[image_name] = np.zeros(self.worker.image_shape, dtype=np.uint16)

            # Resize to standard shape if necessary
            if mask.shape != self.worker.image_shape:
                mask = cv2.resize(mask, (self.worker.image_shape[1], self.worker.image_shape[0]), interpolation=cv2.INTER_NEAREST)

            image_masks_dict[image_name][mask > 0] = label_value

        return image_masks_dict, label_map
    
    def get_mask_by_key(self, mask_key):
        """
        Retrieves a mask from either viewer-defined masks or original data.

        Parameters:
            mask_key (str): Combined image_name and label.

        Returns:
            np.ndarray or None: Binary mask array.
        """
        for viewer in self.gray_viewers:
            if mask_key in viewer.new_mask_dict:
                return viewer.new_mask_dict[mask_key]["mask"]

        for viewer in self.gray_viewers:
            if mask_key in viewer.callback_dict:
                cb_data = viewer.callback_dict[mask_key]
                return (
                    cb_data.get("mask") 
                    or cb_data.get("binary_mask") 
                    or self.reconstruct_mask_from_callback(cb_data)
                )
        return None
    
    def reconstruct_mask_from_callback(self, cb_data):
        """
        Reconstructs a mask from label_mask if direct mask is not present.

        Parameters:
            cb_data (dict): Callback data with image name and label.

        Returns:
            np.ndarray or None: Binary mask.
        """
        image_name = cb_data.get("name")
        label = cb_data.get("label")
        if image_name in self.worker.masks_dict:
            label_mask = self.worker.masks_dict[image_name]["label_mask"]
            return (label_mask == int(label)).astype(np.uint8)
        return None
    
    def assign_label_value(self, label_str, label_map):
        """
        Assigns a unique integer value to a label, used for generating ROIs.

        Parameters:
            label_str (str): Label string.
            label_map (dict): Already assigned label map.

        Returns:
            int: Unique label value.
        """
        if label_str.startswith("drawn_"):
            return 3000 + len(label_map) + 1
        elif '(' in label_str:
            return 1000 + len(label_map) + 1
        try:
            return int(label_str)
        except (ValueError, TypeError):
            return 2000 + len(label_map) + 1
        
    def export_rois_to_zip(self, image_masks_dict, label_map, roi_dir):
        """
        Exports ROIs from masks as .roi files and packages them into a ZIP archive.
        """
        for image_name, full_mask in image_masks_dict.items():
            # Rotate the mask for correct orientation in ImageJ
            rotated_mask = np.rot90(np.flipud(full_mask), k=-1)
            roi_list = []

            for key, label_value in label_map.items():
                # Match keys that belong to this image
                if not key.startswith(image_name + "_"):
                    continue
                # Extract label string
                label = key[len(image_name)+1:]
                # Generate binary mask for this label
                binary_mask = (rotated_mask == label_value).astype(np.uint8)
                # Extract contours (edges of regions)
                contours = measure.find_contours(binary_mask, 0.5)

                for contour in contours:
                    contour = np.round(contour).astype(np.int32)
                    if contour.shape[0] < 10:
                        continue
                    # Create ROI object from contour
                    roi = roifile.ImagejRoi.frompoints(contour)
                    # Name ROI file with merged status if applicable
                    filename = self.generate_roi_filename(image_name, label)
                    roi_list.append((filename, roi))
            # Save all ROIs into a zip file
            zip_path = os.path.join(roi_dir, f"{os.path.splitext(image_name)[0]}.zip")
            if roi_list:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for filename, roi in roi_list:
                        zipf.writestr(filename, roi.tobytes())
            #else:
                #print(f"No ROIs generated for {image_name}")

    def save_rois_to_zip(self, correct_df):
        """
        Converts segmentation masks into ROI files and packages them into ZIP archives.
        """
        active_keys = self.gather_active_keys()
        # Merge active + newly created objects
        combined_df, _ = self.integrate_new_objects(correct_df, active_keys, set())
        combined_df = self.filter_active_labels(combined_df, active_keys)
        # Create directory for storing ROI zips
        roi_dir = self.prepare_roi_dir(
            self.input_form.images_folder_path.text(), 
            self.input_form.roi_folder_name.text()
        )
        # Extract masks and label mappings
        image_masks_dict, label_map = self.collect_masks(combined_df)
        # Export each image’s ROIs to a zip file
        self.export_rois_to_zip(image_masks_dict, label_map, roi_dir)

    def process_merged_masks(self, active_keys, inactive_keys):
        """
        Handles merged masks (groups of regions combined into one).
        """
        new_rows = []
        excluded_rows = []
        merged_keys = set()

        for viewer in self.gray_viewers:
            for group in self.get_merged_groups(viewer):
                group_items = [viewer.get_item_by_id(mid) for mid in group["mask_ids"] if viewer.get_item_by_id(mid)]

                if not group_items:
                    continue

                image_name = group_items[0].name
                labels = sorted(str(item.label) for item in group_items)
                merged_label_str = f"({','.join(labels)})"
                merged_key = f"{image_name}_{merged_label_str}"
                # Record individual keys used in this merged group
                for item in group_items:
                    merged_keys.add(f"{image_name}_{item.label}")

                # Determine whether the group is considered active
                is_active = all(f"{image_name}_{item.label}" in active_keys for item in group_items)
                viewer.callback_dict[merged_key] = {"is_active": is_active}
                if is_active:
                    active_keys.add(merged_key)
                else:
                    inactive_keys.add(merged_key)

                # Create merged mask
                merged_mask = np.zeros(self.worker.image_shape, dtype=np.uint8)
                for item in group_items:
                    merged_mask[item.binary_mask > 0] = 1

                # Store new merged mask
                viewer.new_mask_dict[merged_key] = {
                    "mask": merged_mask,
                    "source": "connect",
                    "label_group": labels,
                    "image_name": image_name,
                }
                self.worker.masks_dict[merged_key] = {"mask": merged_mask}

                # Measure region properties of the merged mask
                intensity_image = self.worker.image_dict.get(image_name)
                df_props = compute_region_properties(merged_mask, intensity_image=intensity_image)
                df_props['label'] = [merged_label_str] * len(df_props)
                df_props['image_name'] = image_name
                df_props['Replicate'] = self.input_form.rep_num.text()
                df_props['Condition'] = self.input_form.condition_name.text()
                

                if is_active:
                    new_rows.append(df_props)
                else:
                    df_props = pixel_conversion(df_props, float(self.input_form.pixel_rate.text()))
                    excluded_rows.append(df_props)

        return new_rows, excluded_rows, merged_keys
    
    def process_disconnected_masks(self, active_keys, existing_keys):
        """
        Processes disconnected masks created by splitting previously merged masks.
        """
        new_rows = []

        for viewer in self.gray_viewers:
            for key, entry in viewer.new_mask_dict.items():
                if entry.get("source") != "disconnect":
                    continue

                label_group = entry.get("label_group", [])
                if not label_group:
                    continue

                image_name = entry["image_name"]
                label = label_group[0]
                cb_key = f"{image_name}_{label}"

                if cb_key not in active_keys or cb_key in existing_keys:
                    continue

                mask = entry["mask"]
                self.worker.masks_dict[key] = {"mask": mask}

                intensity_image = self.worker.image_dict.get(image_name)
                df_props = compute_region_properties(mask, intensity_image=intensity_image)
                df_props['label'] = label
                df_props['image_name'] = image_name
                df_props['Replicate'] = self.input_form.rep_num.text()
                df_props['Condition'] = self.input_form.condition_name.text()
                

                if self.roi_checkbox.isChecked():
                    df_props = self.add_roi_name_column(df_props, is_merged=False)

                new_rows.append(df_props)
        return new_rows

    def process_drawn_masks(self):
        """
        Processes user-drawn masks on the canvas.
        """
        new_rows = []
        for idx, viewer in enumerate(self.gray_viewers):
            items = viewer.mask_items
            image_name = items[0].name if items else f"viewer_{idx}"
            min_area = 100
            # Measure drawn mask regions
            labeled_mask, drawn_props = self.measure_drawn_objects(viewer)

            for prop in drawn_props:
                if prop.area < min_area:
                    continue

                region_mask = (labeled_mask == prop.label).astype(np.uint8)
                drawn_mask_key = f"{image_name}_drawn_{prop.label}"

                # Store the drawn mask
                viewer.new_mask_dict[drawn_mask_key] = {
                    "mask": region_mask,
                    "source": "draw",
                    "image_name": image_name
                }
                self.worker.masks_dict[drawn_mask_key] = {"mask": region_mask}
                viewer.callback_dict[drawn_mask_key] = {
                    "name": image_name,
                    "label": f"drawn_{prop.label}",
                    "is_active": True,
                    "merged": False
                }

                intensity_image = self.worker.image_dict.get(image_name)
                if region_mask.shape != intensity_image.shape:
                    region_mask = cv2.resize(region_mask, (intensity_image.shape[1], intensity_image.shape[0]), interpolation=cv2.INTER_NEAREST)

                labeled_mask = measure.label(region_mask)
                df_drawn_props = compute_region_properties(labeled_mask, intensity_image=intensity_image)
                df_drawn_props['image_name'] = image_name
                df_drawn_props['label'] = f"drawn_{prop.label}"
                df_drawn_props['Replicate'] = self.input_form.rep_num.text()
                df_drawn_props['Condition'] = self.input_form.condition_name.text()
                

                if self.roi_checkbox.isChecked():
                    df_drawn_props = self.add_roi_name_column(df_drawn_props)

                new_rows.append(df_drawn_props)
        return new_rows
    
    def build_combined_and_excluded_df(self, correct_df, new_rows, excluded_rows, inactive_keys, merged_keys):
        """
        Builds combined and excluded DataFrames based on mask status.
        """
        if new_rows:
            new_df = pd.concat(new_rows, ignore_index=True)
            new_df = pixel_conversion(new_df, float(self.input_form.pixel_rate.text()))
            # Remove rows that match excluded labels
            excluded_keys = {
                f"{row['image_name']}_{row['label']}"
                for df in excluded_rows
                for _, row in df.iterrows()
            }
            new_df = new_df[
                ~new_df.apply(lambda r: f"{r['image_name']}_{r['label']}" in excluded_keys, axis=1)
            ]
            combined_df = pd.concat([correct_df, new_df], ignore_index=True)
        else:
            combined_df = correct_df.copy()
        # Build excluded DataFrame
        excluded_df = self.worker.all_props_df[
            (self.worker.all_props_df.apply(lambda r: f"{r['image_name']}_{r['label']}" in inactive_keys, axis=1)) &
            (~self.worker.all_props_df.apply(lambda r: f"{r['image_name']}_{r['label']}" in merged_keys, axis=1))
        ].copy()

        if excluded_rows:
            excluded_df = pd.concat([excluded_df, pd.concat(excluded_rows, ignore_index=True)], ignore_index=True)

        return combined_df, excluded_df

    def build_group_label_map(self):
        """
        Maps individual mask labels to their merged group label string.
        """
        group_map = {}
        for viewer in self.gray_viewers:
            for group in self.get_merged_groups(viewer):
                labels = []
                image_name = None
                for mid in group["mask_ids"]:
                    item = viewer.get_item_by_id(mid)
                    if item:
                        image_name = item.name
                        labels.append(str(item.label))
                if image_name and labels:
                    sorted_labels = sorted(labels, key=int)
                    group_label = f"({','.join(sorted_labels)})"
                    for lbl in sorted_labels:
                        group_map[(image_name, lbl)] = group_label
        return group_map

    def integrate_new_objects(self, correct_df, active_keys, inactive_keys):
        """
        Integrates newly created/edited masks with the existing dataset.
        """
        if not self.gray_viewers:
            return self.worker.all_props_df.copy(), pd.DataFrame()

        all_props_df = self.worker.all_props_df.copy()

         # Filter only active masks---
        correct_df = all_props_df[
        all_props_df.apply(lambda r: f"{r['image_name']}_{r['label']}" in active_keys, axis=1)
        ].copy()

        existing_keys = set(correct_df['image_name'].astype(str) + "_" + correct_df['label'].astype(str))
        self.worker.masks_dict.clear()
        merged_keys = set() # Track keys of individual masks in merged groups
        
        # Process all types of masks
        new_merged_rows, excluded_rows, merged_keys = self.process_merged_masks(active_keys, inactive_keys)
        new_disconnected_rows = self.process_disconnected_masks(active_keys, existing_keys)
        new_drawn_rows = self.process_drawn_masks()

        # Combine everything
        new_rows = new_merged_rows + new_disconnected_rows + new_drawn_rows
        combined_df, excluded_df = self.build_combined_and_excluded_df(correct_df, new_rows, excluded_rows, inactive_keys, merged_keys)

        group_map = self.build_group_label_map()
        merged_individual_keys = set(group_map.keys())

        # Drop rows that are replaced by merged entries
        combined_df = combined_df[
            ~combined_df.apply(
                lambda r: (r['image_name'], str(r['label'])) in merged_individual_keys
                if not str(r['label']).startswith("drawn_") else False,
                axis=1
            )
        ]
        # Do the same for excluded_df (if it's not empty)
        if not excluded_df.empty:
            excluded_df = excluded_df[
                ~excluded_df.apply(
                    lambda r: (str(r['image_name']), str(r['label']).strip()) in merged_individual_keys,
                    axis=1
                )
            ]
        # Final deduplication 
        combined_df = combined_df.drop_duplicates(subset=['image_name', 'label'])
        if not excluded_df.empty:
            excluded_df = excluded_df.drop_duplicates(subset=['image_name', 'label'])

        return combined_df, excluded_df
    
    def measure_drawn_objects(self, viewer):
        """
        Measures regions in the drawing canvas as drawn masks.
        """
        canvas = viewer.drawing_canvas.toImage()
        width, height = canvas.width(), canvas.height()
        ptr = canvas.bits()
        ptr.setsize(height * width * 4)
        arr = np.array(ptr).reshape((height, width, 4))
        # Extract alpha channel (mask)
        alpha = (arr[..., 3] >= 200).astype(np.uint8)

         # Morphological closing to fill gaps
        kernel_close = np.ones((11, 11), np.uint8)
        closed = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel_close) 

        # Extract contours, fill them
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(alpha, dtype=np.uint8)
        cv2.drawContours(filled_mask, contours, -1, 1, thickness=-1)

        # Label filled regions and return their properties
        labeled_mask = measure.label(filled_mask)
        props = measure.regionprops(labeled_mask)
        return labeled_mask, props

    def generate_roi_filename(self, image_name, label):
        base = os.path.splitext(str(image_name))[0]
        label_str = str(label)
        if label_str.startswith("drawn_"):
            return f"{base}_{label_str}.roi"
        elif "_merged" in label_str or "(merged)" in label_str:
            return f"{base}_label_{label_str}_merged.roi"
        elif "_disconnected" in label_str:
            return f"{base}_label_{label_str}_disconnected.roi"
        else:
            return f"{base}_label{label_str}.roi"
        
    def add_roi_name_column(self, df):
        """
        Generates and adds ROI filenames to the DataFrame.
        """
        df['roi_name'] = df.apply(
            lambda r:   self.generate_roi_filename(r['image_name'], r['label'])
            if pd.notnull(r['image_name']) and pd.notnull(r['label']) else "",
            axis=1
        )
        return df
 
    def get_merged_groups(self, viewer):
        """
        Return list of unique active groups based on `connected_groups` only.
        Ensures no stale or unmerged masks are included.
        """
        grouped = []
        for group in viewer.connected_groups:
            group_copy = {
                "mask_ids": set(group["mask_ids"]),
                "color": group["color"],
                "mask": group.get("mask")
            }
            grouped.append(group_copy)

        return grouped

    def start_processing(self):
        """
        Initializes and starts image processing workflow:
        - Clears previous viewers, masks, and UI elements.
        - Reads all form input values.
        - Sets up and starts a background worker thread.
        """
        # Clear previous image viewers and their associated data
        if hasattr(self, "gray_viewers"):
            for viewer in self.gray_viewers:
                viewer.connected_groups.clear()
                if hasattr(viewer, "mask_id_to_group"):
                    viewer.mask_id_to_group.clear()
                if hasattr(viewer, "new_mask_dict"):
                    viewer.new_mask_dict.clear()
            self.gray_viewers.clear()

        # Clear image layout in GUI
        if self.image_layout is not None:
            while self.image_layout.count():
                child =  self.image_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        self.image_layout = None  # Will be recreated during `create_images_tab()`

        # Clear previous worker and data
        if hasattr(self, "worker"):
            self.worker.masks_dict.clear()
            self.worker.image_dict.clear()
            del self.worker 
        # Retrieve input from form
        try:
            folder_path = self.input_form.images_folder_path.text()
            csv_file_name = self.input_form.csv_file_name.text()
            roi_folder_name = self.input_form.roi_folder_name.text()
            condition_name = self.input_form.condition_name.text()
            rep_num = self.input_form.rep_num.text()
            main_marker_identifier = self.input_form.unique_main_marker_identifier.text()
            nucleus_identifier = self.input_form.unique_nucleus_identifier.text()
            color = self.input_form.main_marker_channel_dropdown.currentText()

            # Get slider values for contrast ranges
            main_marker_contrast_low = self.input_form.main_marker_low_contrast_slider.value()
            main_marker_contrast_high = self.input_form.main_marker_high_contrast_slider.value()
            nucleus_contrast_low = self.input_form.nucleus_low_contrast_slider.value()
            nucleus_contrast_high = self.input_form.nucleus_high_contrast_slider.value()

            # Get numeric parameters
            diam = float(self.input_form.diameter.text())
            flow_thresh = float(self.input_form.flow_threshold.text())
            min_area = float(self.input_form.min_area.text())
            min_non_black_pixels_percentage = float(self.input_form.min_non_black_pixels_percentage.text())
            intensity_threshold = float(self.input_form.intensity_threshold.text())
            min_nucleus_pixels_percentage = float(self.input_form.min_nucleus_pixels_percentage.text())
            nucleus_pixel_threshold = float(self.input_form.nucleus_pixel_threshold.text())
            pixel_conv_rate = None
            pixel_conv_rate_text = self.input_form.pixel_rate.text()
            # Loading images from folder
            self.images = open_folder(folder_path, [main_marker_identifier,nucleus_identifier])
            if pixel_conv_rate_text != "":
                try:
                    pixel_conv_rate = float(pixel_conv_rate_text)
                except ValueError:
                    self.update_status_label(f"Invalid input, please refer to instructions!")
                    self.processing_in_progress = False
                    self.input_form.process_button.setEnabled(True)  # Re-enable the start button
                    return

            nucleus_channel_present = self.input_form.nucleus_checkbox.isChecked()

            # Creating the worker to process images in the background
            self.worker = ImageProcessingWorker(self.images, folder_path, condition_name, rep_num, main_marker_identifier, nucleus_identifier, color, 
                                                main_marker_contrast_low, main_marker_contrast_high, nucleus_contrast_low, nucleus_contrast_high, 
                                                diam, flow_thresh, min_area, min_non_black_pixels_percentage, intensity_threshold, min_nucleus_pixels_percentage,
                                            nucleus_pixel_threshold, pixel_conv_rate, csv_file_name, roi_folder_name, 
                                            self.input_form.progress_bar, self.model, nucleus_channel_present=nucleus_channel_present)

            # Connecting the worker's signal to the slot to update the UI
            self.worker.status_update.connect(self.update_status_label)
            self.worker.image_processed.connect(self.add_images_to_scrollable_area)
            self.worker.show_save_all.connect(self.show_save_all)
            self.worker.finished_processing.connect(self.processing_done)
            self.worker.progress_updated.connect(self.update_progress)
            # Starting the worker in the background
            self.worker.start()
            
        except ValueError as e:
            # Showing an error message if invalid input
            self.update_status_label(f"Invalid input, please refer to instructions!")
            self.processing_in_progress = False
            self.input_form.process_button.setEnabled(True)  # Re-enabling the start button

    def update_status_label(self, message):
        """
        Updates the processing status message in the UI.
        """
        self.input_form.processing_label.setText(message)
        
    def show_save_all(self):
        self.image_layout.addWidget(self.button_widget)

    def processing_done(self):
        """
        Called when processing is completed. Re-enables UI components.
        """
        self.processing_in_progress = False
        self.input_form.process_button.setEnabled(True)
     
    def update_progress(self, value):
        """
        Updates the progress bar based on processing progress.

        Args:
            value (int): Percent progress value (0-100)
        """
        self.input_form.progress_bar.setValue(value)  # Updating bar with percentage
        self.input_form.progress_bar.setFormat(f"{value}%")  # Display percentage
        self.input_form.progress_bar.repaint()  # Force UI update

    def add_images_to_scrollable_area(self, title, pixmap_gray, pixmap_rgb, pixmap_overlay, masks_list):
        """
        Adds a processed image set (grayscale, RGB, overlay) with labeled masks into the scroll area.

        Args:
            title (str): Image name or label
            pixmap_gray (QPixmap): Grayscale image
            pixmap_rgb (QPixmap): RGB image
            pixmap_overlay (QPixmap): Overlay image with segmentation
            masks_list (list): Segmentation masks for ImageViewer
        """
        container = QWidget()
        layout = QHBoxLayout()

        # Image display widgets
        label_gray = QLabel()
        label_rgb = QLabel()
        label_overlay = QLabel()
        label_rgb.setScaledContents(True)
        label_overlay.setScaledContents(True)
        label_gray.setScaledContents(True)

        # Resize and set scaled images
        scaled_width = 430 # Target width for each image
        scaled_height = 530  # Target height for each image
        scaled_pixmap_gray = pixmap_gray.scaled(scaled_width, scaled_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        scaled_pixmap_rgb = pixmap_rgb.scaled(scaled_width, scaled_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        scaled_pixmap_overlay = pixmap_overlay.scaled(scaled_width, scaled_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        label_rgb.setPixmap(scaled_pixmap_rgb)
        label_overlay.setPixmap(scaled_pixmap_overlay)
        label_gray.setPixmap(scaled_pixmap_gray)

        # Setup image viewer for grayscale/masks interaction
        show_labels = self.input_form.cell_labels_checkbox.isChecked()
        font_size = self.input_form.cell_label_font_size_spinbox.value()
        # Using the scaled gray image in ImageViewer
        if not hasattr(self, "viewer_mode_controller"):
            self.viewer_mode_controller = ViewerModeController()
        controller = self.viewer_mode_controller    
        self.gray_viewer = ImageViewer(scaled_pixmap_gray, masks_list, font_size, show_labels, title, mode_controller=controller)

        # Add to layout
        layout.addWidget(label_rgb)
        layout.addWidget(label_overlay)
        layout.addWidget(self.gray_viewer)
        container.setLayout(layout)

        self.gray_viewers.append(self.gray_viewer)

        # Creating a combined container for the title and image container
        combined_container = QWidget()
        combined_layout = QVBoxLayout()
        combined_layout.setSpacing(10)  
        combined_layout.addWidget(QLabel(f"<b><span style='font-size: 22px;'>{title}</span></b>"))
        combined_layout.addWidget(container)
        combined_container.setLayout(combined_layout)
        
        if self.image_layout is None:
             self.image_layout = self.create_images_tab()

        self.image_layout.addWidget(combined_container, alignment=Qt.AlignmentFlag.AlignTop)

    def mousePressEvent(self, event):
        """
        Close help text when clicking anywhere
        """
        if self.help_text.isVisible():
            self.help_text.setVisible(False)
        event.accept()

    def closeEvent(self, event):
        """
        Override the closeEvent to close the help window when the main window is closed
        """
        if self.help_text.isVisible():
            self.help_text.setVisible(False)  # Close help window before closing main window
        event.accept()  # Proceed with closing the main window


class ImageProcessingWorker(QThread):
    """
    QThread subclass for processing microscopy images using Cellpose segmentation.
    Emits progress and status updates for integration with a PyQt UI.
    """
    # === Signals ===
    image_processed = pyqtSignal(str, QPixmap, QPixmap, QPixmap, list) # Emits name, grayscale, RGB, overlay, and mask list
    status_update = pyqtSignal(str)  # New signal for status updates
    show_save_all = pyqtSignal() # Signal to show the save all button
    finished_processing = pyqtSignal() # Signal for the end of the process
    progress_updated = pyqtSignal(int) # Signal for a progress bar updates
    
    def __init__(self, images, folder_path, condition_name, rep_num, main_marker_identifier, nucleus_identifier,  color,  main_marker_contrast_low,
                 main_marker_contrast_high, nucleus_contrast_low, nucleus_contrast_high,  diam, thresh, min_area, min_non_black_pixels_percentage,
                 intensity_threshold, min_nucleus_pixels_percentage, nucleus_pixel_threshold, pixel_conv_rate, csv_file_name, roi_folder_name, progress_bar, model,
                 nucleus_channel_present=True):
        """
        Initializes the worker thread with parameters for image analysis.
        """
        super().__init__()
        self.folder_path = folder_path
        self.images = images
        self.rep_num = rep_num
        self.condition_name = condition_name
        self.main_marker_identifier = main_marker_identifier
        self.nucleus_identifier = nucleus_identifier
        self.color = color
        self.main_marker_contrast_low = main_marker_contrast_low
        self.main_marker_contrast_high = main_marker_contrast_high
        self.nucleus_contrast_low = nucleus_contrast_low
        self.nucleus_contrast_high = nucleus_contrast_high
        self.diam = diam
        self.thresh = thresh
        self.min_area = min_area
        self.min_non_black_pixels_percentage = min_non_black_pixels_percentage
        self.intensity_threshold = intensity_threshold
        self.min_nucleus_pixels_percentage = min_nucleus_pixels_percentage
        self.nucleus_pixel_threshold = nucleus_pixel_threshold
        self.pixel_conv_rate = pixel_conv_rate
        self.csv_file_name = csv_file_name
        self.roi_folder_name = roi_folder_name
        self.active = True
        self.count = 0  # Count of successfully processed images
        self.progress_bar = progress_bar
        self.model = model
        self.nucleus_channel_present = nucleus_channel_present
        self.image_dict = {} # Stores preprocessed main marker images
        self.masks_dict = {} # Stores predicted masks keyed by image+label

    def stop(self):
        """
        Stop processing (used by main UI).
        """
        self.active = False
        
    def run(self):
        """
        Main execution logic for the thread.
        Performs image preprocessing, segmentation, analysis, and emits results.
        """
         # === Preliminary checks ===
    
        if not self.active:  # Stop processing if active flag is false
            return
        
        # Checking if the folder path is empty or invalid
        if not self.folder_path or not os.path.isdir(self.folder_path):
            self.status_update.emit("Invalid folder path!")
            self.finished_processing.emit() 
            return
        
        # Checking if the selected color is empty (i.e., not selected)
        if self.condition_name == "":
            self.status_update.emit("Please input a condition name!")
            self.finished_processing.emit() 
            return 
        
        if self.rep_num == "":
            self.status_update.emit("Please input a replicate number!")
            self.finished_processing.emit() 
            return\
        
        main_marker_channel_value = self.color
        if main_marker_channel_value == "":
            self.status_update.emit("Please select a valid segmentation marker channel color!")
            self.finished_processing.emit() 
            return
    
        if  not self.pixel_conv_rate:
            self.status_update.emit("Please input pixel-to-micron conversion rate")
            self.finished_processing.emit() 
            return
        
        if self.pixel_conv_rate > 2:
            self.status_update.emit("Invalid pixel-to-micron conversion rate!")
            self.finished_processing.emit() 
            return
        
        # Ensuring 'self.images' is defined and contains the expected data
        if not hasattr(self, 'images') or not self.images:
            self.status_update.emit("No images loaded into the system!")
            self.finished_processing.emit() 
            return
        
        # Count only images matching the main marker identifier
        total_images = sum(1 for key in self.images if self.main_marker_identifier in key)
        if total_images == 0:
            self.status_update.emit("No images found in the folder. Check file IDs")
            self.finished_processing.emit() 
            return
        
        # === Begin processing ===
        self.status_update.emit("Processing started...")
        self.all_props_df = pd.DataFrame()
        num_images = len(self.images)  # Getting total number of images
        self.masks_dict = {}  # Dictionary to store masks

        fail = True # Indicates whether any images were processed
        for num, (name, image) in enumerate(self.images.items()):
            if not self.active:
                break

            if self.main_marker_identifier in name and self.main_marker_identifier!="" and self.active:
                main_marker_image_name = name
                main_marker_image_path = image
                # Load image shape for future use
                if not hasattr(self, "image_shape"):
                    image_array = np.array (Image.open(main_marker_image_path))
                    self.image_shape = image_array.shape
                # === Handle nucleus channel ===
                if self.nucleus_channel_present:
                    nucleus_name = name.replace(self.main_marker_identifier, self.nucleus_identifier)
                    if (nucleus_name not in self.images or self.nucleus_identifier=="") and (self.active):  # Prevent KeyError
                        self.status_update.emit(f"Missing nucleus image: {nucleus_name}")
                        self.finished_processing.emit()
                        return
                    nucleus_image_path = self.images[nucleus_name]
 
                if self.active:
                    try:
                        # === Preprocessing ===
                        if not self.active:
                            break
                        if self.nucleus_channel_present:
                            main_marker_image, nucleus_image, diamet, marker_channel_color, rgb = image_preprocessing(main_marker_image_path, nucleus_image_path=nucleus_image_path if self.nucleus_channel_present else None,
                                                                                                        main_marker_channel = self.color,
                                                                                                        main_marker_contrast_high = self.main_marker_contrast_high,main_marker_contrast_low = self.main_marker_contrast_low,
                                                                                                        nucleus_contrast_low = self.nucleus_contrast_low, nucleus_contrast_high = self.nucleus_contrast_high, # image of the channel that will be used for segmentation purposes (red/actin channel used here)
                                                                                                        min_non_black_pixels_percentage = self.min_non_black_pixels_percentage,
                                                                                                        intensity_threshold=self.intensity_threshold,  pixel_conv_rate=self.pixel_conv_rate,
                                                                                                        diam = self.diam,
                                                                                                        nucleus_channel_present=self.nucleus_channel_present)
                        else:
                            main_marker_image, diamet, marker_channel_color, rgb = image_preprocessing(main_marker_image_path, nucleus_image_path=nucleus_image_path if self.nucleus_channel_present else None,
                                                                                                        main_marker_channel = self.color,
                                                                                                        main_marker_contrast_high = self.main_marker_contrast_high,main_marker_contrast_low = self.main_marker_contrast_low,
                                                                                                        nucleus_contrast_low = self.nucleus_contrast_low, nucleus_contrast_high = self.nucleus_contrast_high, # image of the channel that will be used for segmentation purposes (red/actin channel used here)
                                                                                                        min_non_black_pixels_percentage = self.min_non_black_pixels_percentage,
                                                                                                        intensity_threshold=self.intensity_threshold,  pixel_conv_rate=self.pixel_conv_rate,
                                                                                                        diam = self.diam,
                                                                                                        nucleus_channel_present=self.nucleus_channel_present)
                        if not self.active:
                            break
                        self.image_dict[main_marker_image_name] = main_marker_image.copy()

                        # === Run segmentation model ===
                        predicted_masks, _, _ = self.model.eval(main_marker_image, diameter=diamet, flow_threshold = self.thresh,  channels=[0, marker_channel_color])

                        if not self.active:
                            break

                        # === Analyze segmented cells ===
                        df, overlay_image, gray_image, masks_list = analyze_segmented_cells(predicted_masks, main_marker_image, main_marker_image_name, 
                                                                                        pixel_conv_rate=self.pixel_conv_rate,
                                                                                        rgb_image = rgb, min_area = self.min_area,
                                                                                        condition_name= self.condition_name, replicate_num = self.rep_num,
                                                                                        nucleus_image=nucleus_image if self.nucleus_channel_present else None,
                                                                                        min_nucleus_pixels_percentage=self.min_nucleus_pixels_percentage if self.nucleus_channel_present else None,
                                                                                        nucleus_pixel_threshold=self.nucleus_pixel_threshold if self.nucleus_channel_present else None,
                                                                                        nucleus_channel_present=self.nucleus_channel_present)
                        if not self.active:
                            break
                        # Store masks in a dictionary
                        for mask, label in zip(masks_list, df['label']):
                            mask_key = f"{main_marker_image_name}{label}"
                            self.masks_dict[mask_key] = mask

                        # === Normalize images and emit UI updates ===
                        if df is not None and self.active:
                            self.all_props_df = pd.concat([self.all_props_df, df], ignore_index=True)
                            # Normalizing images to uint8
                            if not self.active:
                                break
                            gray_image = normalize_to_uint8(gray_image)
                            rgb = normalize_to_uint8(rgb)  # Only normalizing if it's not in the range 0–255 already
                            overlay_image = normalize_to_uint8(overlay_image)

                            if not self.active:
                                break
                            # Converting processed images to QPixmap
                            pixmap_gray = convert_to_pixmap(gray_image, QImage.Format.Format_Grayscale8)
                            pixmap_rgb = convert_to_pixmap(np.concatenate([rgb, np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)], axis=-1), QImage.Format.Format_RGBA8888)
                            pixmap_overlay = convert_to_pixmap(np.concatenate([overlay_image, np.full((overlay_image.shape[0], overlay_image.shape[1], 1), 255, dtype=np.uint8)], axis=-1), QImage.Format.Format_RGBA8888)
                            # Emitting signal to update the UI with the processed images

                            if len(df)==1:
                                display_name = f"{main_marker_image_name}: 1 cell processed"
                            else:
                                display_name = f"{main_marker_image_name}: {len(df)} cells processed"

                            self.image_processed.emit(display_name, pixmap_gray, pixmap_rgb, pixmap_overlay, masks_list)

                            self.count += 1

                    except Exception as e:
                        #print(e)
                        continue  # Moving to the next image
             # === Update progress ===
            self.progress_updated.emit(int(((num + 1) / num_images) * 100))
        
        # === Post-processing ===
        if (fail and self.active and self.count >= 1) or (not self.active):
            self.all_props_df = pixel_conversion(self.all_props_df, self.pixel_conv_rate)
            self.status_update.emit(f"Processing completed! {self.count} images processed.")
            self.show_save_all.emit()
            self.finished_processing.emit() 
        
        elif fail:
            self.status_update.emit("No images processed. Please check your input parameters.")
            self.finished_processing.emit() 

def set_light_palette(app):
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Base, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
    app.setPalette(palette)
    
 # === Main execution for the app ===
if __name__ == "__main__":
    
    if not sys.stdout:
        sys.stdout = open(os.devnull, "w")
    if not sys.stderr:
        sys.stderr = open(os.devnull, "w")
    
    #disable_tqdm = not sys.stdout or not sys.stdout.isatty()
    #for x in tqdm(range(100), disable=disable_tqdm):
    #    pass
    
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(resource_path_2("icons/icon.ico")))
    app.setStyle(QStyleFactory.create("Fusion"))
    # Optional: your global stylesheet
    set_light_palette(app)
    app.setStyleSheet("""
        QWidget {
            background-color: white;
            color: black;
        }
        QPushButton {
            background-color: #e0e0e0;
            border: 1px solid #a0a0a0;
        }
    """)
    main_window = ImageProcessingApp()
    main_window.show()
    sys.exit(app.exec())

