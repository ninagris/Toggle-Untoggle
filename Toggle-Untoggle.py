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

from skimage import measure
from cellpose import models
from PIL import Image

from PyQt6.QtWidgets import QApplication, QLabel, QPushButton
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QMainWindow
from PyQt6.QtWidgets import QTabWidget, QScrollArea, QSizePolicy, QStyleFactory, QCheckBox
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer,QSize

from image_analysis_pipeline import open_folder, image_preprocessing, analyze_segmented_cells, convert_to_pixmap, normalize_to_uint8, pixel_conversion, compute_region_properties
from toggle import ImageViewer, ViewerModeController
from input_form_components import InputFormWidget, DraggableTextEdit

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = models.CellposeModel(gpu=True, model_type='cyto3')
        self.model.device = torch.device("mps")  # Force MPS usage (for mac gpu use)

        self.setWindowTitle("Toggle-Untoggle")
        self.setWindowIcon(QIcon("icon.png"))
        self.resize(850, 700)
        self.setMinimumSize(850, 700)
        self.showFullScreen()
        self.viewer_mode_controller = ViewerModeController()
        # Main tab widget With tabs on top
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

        # Creating the clickable GitHub link QLabel
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
        self.help_text = DraggableTextEdit(parent=self)
        self.help_text.setVisible(False)
        font = QFont("Arial", 18) 
        self.help_text.setFont(font)
        self.help_text.setReadOnly(True)
        # Customizing Tab for the input parameters
        self.input_tab = QWidget()       # Create the tab widget
        self.input_tab.setLayout(QVBoxLayout())
        self.input_form = InputFormWidget(parent=self.input_tab, help_text=self.help_text, tab_widget=self.tabs)
        self.input_tab.layout().addWidget(self.input_form)
        self.tabs.addTab(self.input_tab, "Input Parameters")
        # List for storing grayscale images with interactive masks
        self.gray_viewers = []
        self.processing_in_progress = False
        self.input_form.processClicked.connect(self.on_process_clicked)
        self.input_form.saveClicked.connect(self.collect_all_callbacks)
        self.images_tab = None  # Start without images tab
        self.image_layout = None  # This will hold image layout
    
    def update_modes(self):
        mode = None  # Initialize mode
        sender = self.sender()
        if sender == self.toggle_checkbox and sender.isChecked():
            self.correction_checkbox.setChecked(False)
            self.drawing_checkbox.setChecked(False)
            self.erase_checkbox.setChecked(False)
            mode = "toggle"
        elif sender == self.correction_checkbox and sender.isChecked():
            self.toggle_checkbox.setChecked(False)
            self.drawing_checkbox.setChecked(False)
            self.erase_checkbox.setChecked(False)
            mode = "connect"
        elif sender == self.drawing_checkbox and sender.isChecked():
            self.toggle_checkbox.setChecked(False)
            self.correction_checkbox.setChecked(False)
            self.erase_checkbox.setChecked(False)
            mode = 'draw'
        elif sender == self.erase_checkbox and sender.isChecked():
            self.toggle_checkbox.setChecked(False)
            self.correction_checkbox.setChecked(False)
            self.drawing_checkbox.setChecked(False)
            mode = 'erase'

        self.viewer_mode_controller.set_mode(mode)  
    
    def handle_model_selection(self, model_type: str, custom_path: str):

        # Always clear the current model first
        self.model = None
        try:
            if model_type == "" and not custom_path.strip():
                raise ValueError("Please input a valid path to the custom model.")

            if custom_path:
                if not os.path.exists(custom_path):
                    raise ValueError("Custom model path does not exist.")
                self.model = models.CellposeModel(gpu=True, pretrained_model=custom_path)

            elif model_type in ["cyto", "cyto2", "cyto3", "nuclei"]:
                self.model = models.CellposeModel(gpu=True, model_type=model_type)

            else:
                raise ValueError("Invalid model type selection.")

            self.model.device = torch.device("mps")

        except Exception as e:
            self.model = None  # ← extra safety
            print(f"[MODEL] Model loading failed: {e}")
            raise

    def resizeEvent(self, event):
        # Your aspect ratio code here (keep as is)
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

        if hasattr(self, "help_text") and self.help_text.isVisible():
            # Defer centering so all layouts & sizes update first
            QTimer.singleShot(0, self.help_text.center_in_parent)

    def on_process_stop(self):
        """
        Enabling the button to stop the process
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
        if self.processing_in_progress:
            return
        
        if hasattr(self, "gray_viewers"):
            for viewer in self.gray_viewers:
                viewer.connected_groups.clear()
                if hasattr(viewer, "mask_id_to_group"):
                    viewer.mask_id_to_group.clear()
                if hasattr(viewer, "new_mask_dict"):
                    viewer.new_mask_dict.clear()
            self.gray_viewers.clear()  # Remove old viewers

        try:
            # Get the current values live from the widget
            current_model_text = self.input_form.model_selector_widget.model_dropdown.currentText()

            if current_model_text == "custom model":
                model_type = ""
                custom_path = self.model_selector_widget.custom_model_input.text().strip()
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
        self.drawing_checkbox.setIcon(QIcon("icons/pen.png"))  # path to your pen icon
        self.drawing_checkbox.setIconSize(QSize(25, 25))  # adjust icon size
        self.drawing_checkbox.setText("")
        self.erase_checkbox = QCheckBox()
        self.erase_checkbox .setIcon(QIcon("icons/eraser.png"))  # path to your pen icon
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
        all_props_df = self.worker.all_props_df.copy()
        all_props_df['combined_key'] = all_props_df['image_name'].astype(str) + '_' + all_props_df['label'].astype(str)

        # Build lookup of all inactive keys
        active_keys = set()
        inactive_keys = set()

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
        # === Step 1: Get currently active (toggled-on) objects from viewers ===
        active_keys = set()
        inactive_keys = set()

        for viewer in self.gray_viewers:
            for cb_key, cb_data in viewer.callback_dict.items():
                if not cb_data.get("is_active"):
                    inactive_keys.add(cb_key)
                    continue
                active_keys.add(cb_key)

        # === Step 2: Combine all active + newly merged/drawn/disconnected ===
        combined_df, excluded_df = self.integrate_new_objects(correct_df, active_keys, inactive_keys)
        if self.roi_checkbox.isChecked():
            combined_df = self.add_roi_name_column(combined_df)
            excluded_df = self.add_roi_name_column(excluded_df)

     
        output_dir =self.input_form.images_folder_path.text()
        output_csv = os.path.join(output_dir, self.input_form.csv_file_name.text() + '.csv')
        combined_df.to_csv(output_csv, index=False)

        excluded_csv = os.path.join(output_dir, 'excluded_objects.csv')
  
        if not excluded_df.empty:
            excluded_df.to_csv(excluded_csv, index=False)
        else:
            if os.path.exists(excluded_csv):
                os.remove(excluded_csv)


    def save_rois_to_zip(self, correct_df):
        """
        Converts segmentation masks into ROI files and packages them into ZIP archives.
        
        Parameters:
            correct_df (pd.DataFrame): DataFrame containing valid objects to convert into ROIs.
        """
        # Step 1: Gather active keys from all viewers
        active_keys = set()
        for viewer in self.gray_viewers:
            for cb_key, cb_data in viewer.callback_dict.items():
                if cb_data.get("is_active", False):
                    active_keys.add(cb_key)

        # Step 2: Integrate new objects to get combined_df with active masks
        combined_df, _ = self.integrate_new_objects(correct_df, active_keys, set())

        # Step 3: Filter combined_df to only include active masks with valid labels
        combined_df = combined_df[
            combined_df.apply(
                lambda r: f"{r['image_name']}_{r['label']}" in active_keys and pd.notnull(r['label']) and str(r['label']).strip() != '',
                axis=1
            )
        ]
        # Step 4: Create ROI directory and prepare image masks
        roi_dir = os.path.join(self.input_form.images_folder_path.text(), self.input_form.roi_folder_name.text())

        # ✅ Delete existing ROI folder if it exists
        if os.path.exists(roi_dir):
            shutil.rmtree(roi_dir)

        # Then create a fresh directory
        os.makedirs(roi_dir, exist_ok=True)
        image_masks_dict = {}

        # Step 5: Process active masks (individual and merged)
        label_map = {}
        image_masks_dict = {}
      
        for _, row in combined_df.iterrows():
            image_name = row['image_name']
            label = row['label']
            label_str = str(label)
            mask_key = f"{image_name}_{label_str}"
            

            mask = None

            # === 1. Try to get mask from viewer.new_mask_dict (merged masks) ===
            for viewer in self.gray_viewers:
                if mask_key in viewer.new_mask_dict:
                    mask = viewer.new_mask_dict[mask_key]["mask"]
                    break

            # === 2. Try to get mask from viewer.callback_dict and reconstruct from worker ===
            if mask is None:
                for viewer in self.gray_viewers:
                    if mask_key in viewer.callback_dict:
                        cb_data = viewer.callback_dict[mask_key]

                        # Try direct mask in callback
                        if "mask" in cb_data:
                            mask = cb_data["mask"]
                            break
                        elif "binary_mask" in cb_data:
                            mask = cb_data["binary_mask"]
                            break
                        else:
                            # Reconstruct from label_mask using cb_data
                            image_name_cb = cb_data.get("name")
                            label_cb = cb_data.get("label")

                            if image_name_cb in self.worker.masks_dict:
                                label_mask = self.worker.masks_dict[image_name_cb]["label_mask"]
                                mask = (label_mask == int(label_cb)).astype(np.uint8)
                                break

            # === 3. Skip if mask still not found or empty ===
            if mask is None:
                continue
            if np.sum(mask) == 0:
                continue

            # === 4. Assign label value
            if label_str.startswith("drawn_"):
                label_value = 3000 + len(label_map) + 1
            elif '(' in label_str:
                label_value = 1000 + len(label_map) + 1
            else:
                try:
                    label_value = int(label)
                except (ValueError, TypeError):
                    label_value = 2000 + len(label_map) + 1

           
            label_map[mask_key] = label_value

            # === 5. Store mask
            if image_name not in image_masks_dict:
                shape = self.worker.image_shape
                image_masks_dict[image_name] = np.zeros(shape, dtype=np.uint16)

            if image_name not in image_masks_dict:
                    shape = self.worker.image_shape
                    image_masks_dict[image_name] = np.zeros(shape, dtype=np.uint16)

            # ✅ Resize if needed
            if mask.shape != self.worker.image_shape:
                mask = cv2.resize(mask, (self.worker.image_shape[1], self.worker.image_shape[0]), interpolation=cv2.INTER_NEAREST)

            image_masks_dict[image_name][mask > 0] = label_value
    
        # Step 6: Generate ROI files for each image
        for image_name, full_mask in image_masks_dict.items():
            rotated_mask = np.rot90(np.flipud(full_mask), k=-1)
            roi_list = []

            labels = []
            for key in label_map:
                if key.startswith(image_name + "_"):
                    labels.append(key[len(image_name)+1:])


            for label in labels:
                mask_key = f"{image_name}_{label}"
                label_value = label_map.get(mask_key)
            
                if label_value is None:
                    continue

                binary_mask = (rotated_mask == label_value).astype(np.uint8)
               
                contours = measure.find_contours(binary_mask, 0.5)

                for contour in contours:
                    contour = np.round(contour).astype(np.int32)
                    if contour.shape[0] < 10:
                        continue
                    roi = roifile.ImagejRoi.frompoints(contour)
                    roi_filename = f"{os.path.splitext(image_name)[0]}_label{label}{'_merged' if '(' in label else ''}.roi"
                    roi_list.append((roi_filename, roi))

                    # Step 7: Save ROIs to ZIP file
                    zip_path = os.path.join(roi_dir, f"{os.path.splitext(image_name)[0]}.zip")
                    if roi_list:  # Check if roi_list is non-empty
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for roi_filename, roi in roi_list:
                                zipf.writestr(roi_filename, roi.tobytes())
                    else:
                        print(f"No ROIs generated for {image_name}")


        # === Remove inactive + reused label rows ===
    def integrate_new_objects(self, correct_df, active_keys, inactive_keys):
        if not self.gray_viewers:
            return self.worker.all_props_df.copy(), pd.DataFrame()

        # Start with a copy of all_props_df
        all_props_df = self.worker.all_props_df.copy()

        # --- Step 1: Keep only active rows in correct_df ---
        correct_df = all_props_df[
        all_props_df.apply(lambda r: f"{r['image_name']}_{r['label']}" in active_keys, axis=1)
        ].copy()
       
        existing_keys = set(correct_df['image_name'].astype(str) + "_" + correct_df['label'].astype(str))
        new_rows = []
        excluded_rows = []
        self.worker.masks_dict.clear()
        merged_keys = set() # Track keys of individual masks in merged groups

        # --- Step 3: Add merged group masks ---
        for viewer in self.gray_viewers:
            for group in self.get_merged_groups(viewer):
                group_items = [
                viewer.get_item_by_id(mid)
                for mid in group["mask_ids"]
                if viewer.get_item_by_id(mid)
            ]

                if not group_items:
                    continue  # Still skip if nothing was found at all

                image_name = group_items[0].name
                labels = sorted(str(item.label) for item in group_items)
                merged_label_str = f"({','.join(labels)})"
                merged_key = f"{image_name}_{merged_label_str}"

                # Add individual mask keys to merged_keys
                for item in group_items:
                    merged_keys.add(f"{image_name}_{item.label}")

                # Determine activity based on constituent masks
                is_active = all(f"{image_name}_{item.label}" in active_keys for item in group_items)

                # Update callback_dict
                viewer.callback_dict[merged_key] = {"is_active": is_active}
                if is_active:
                    active_keys.add(merged_key)
                else:
                    inactive_keys.add(merged_key)

                # Build merged mask regardless of activity
                merged_mask = np.zeros(self.worker.image_shape, dtype=np.uint8)
                for item in group_items:
                    merged_mask[item.binary_mask > 0] = 1

                viewer.new_mask_dict[merged_key] = {
                    "mask": merged_mask,
                    "source": "connect",
                    "label_group": labels,
                    "image_name": image_name,
                }
                self.worker.masks_dict[merged_key] = {"mask": merged_mask}

                # Compute properties
                intensity_image = self.worker.image_dict.get(image_name)
                df_props = compute_region_properties(merged_mask, intensity_image=intensity_image)

                # Overwrite label properly
                df_props['label'] = [merged_label_str] * len(df_props)
                df_props['image_name'] = image_name
                df_props['Replicate'] = self.input_form.condition_name.text()
                df_props['Condition'] = self.input_form.rep_num.text()



                # Save to new or excluded
                if is_active:
                    new_rows.append(df_props)
                else:
                    df_props = pixel_conversion(df_props, float(self.input_form.pixel_rate.text()))
                    excluded_rows.append(df_props)

        # --- Step 4: Add disconnected (ungrouped) masks ---
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
                df_props['Replicate'] = self.input_form.condition_name.text()
                df_props['Condition'] = self.input_form.rep_num.text()


                if self.roi_checkbox.isChecked():  # or your actual checkbox variable
                    df_props = self.add_roi_name_column(df_props, is_merged=True)  # or is_disconnected=True or False depending on case

                new_rows.append(df_props)
        
         # --- Step 5: Add drawn masks ---
        for idx, viewer in enumerate(self.gray_viewers):
            items = viewer.mask_items
            image_name = items[0].name if items else f"viewer_{idx}"


            canvas = viewer.drawing_canvas.toImage()
            width, height = canvas.width(), canvas.height()
            ptr = canvas.bits()
            ptr.setsize(height * width * 4)
            arr = np.array(ptr).reshape((height, width, 4))

            alpha = (arr[..., 3] >= 200).astype(np.uint8)


            kernel_close = np.ones((11, 11), np.uint8)
            closed = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel_close) 

            # === Step 2: Find contours on filled mask ===
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # === Step 3: Fill contours ===
            filled_mask = np.zeros_like(alpha, dtype=np.uint8)
            cv2.drawContours(filled_mask, contours, -1, 1, thickness=-1)  # binary mask (1s)

            # === Step 4: Label and filter regions ===
            labeled_mask = measure.label(filled_mask)
            props = measure.regionprops(labeled_mask)

            min_area = 100  # Filter out noise
            for prop in props:
                if prop.area < min_area:
                    continue

                region_mask = (labeled_mask == prop.label).astype(np.uint8)
                drawn_mask_key = f"{image_name}_drawn_{prop.label}"

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
                # === Get intensity image ===
                intensity_image = self.worker.image_dict.get(image_name)

               # Make sure intensity_image is grayscale
                if intensity_image.ndim == 3:
                    intensity_image = cv2.cvtColor(intensity_image, cv2.COLOR_RGB2GRAY)

                # Resize the mask instead of the image
                if region_mask.shape != intensity_image.shape:
                    region_mask = cv2.resize(
                        region_mask,
                        (intensity_image.shape[1], intensity_image.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                        )

                labeled_mask = measure.label(region_mask)
                df_drawn_props = compute_region_properties(labeled_mask, intensity_image=intensity_image)
               

                df_drawn_props['image_name'] = image_name
                df_drawn_props['label'] = f"drawn_{prop.label}"
                df_drawn_props['Replicate'] = self.input_form.condition_name.text()
                df_drawn_props['Condition'] = self.input_form.rep_num.text()

                if self.roi_checkbox.isChecked():
                    df_drawn_props = self.add_roi_name_column(df_drawn_props)

                new_rows.append(df_drawn_props)

        # --- Step 5: Update all_props_df with new rows ---
        if new_rows:
            new_df = pd.concat(new_rows, ignore_index=True)
            new_df = pixel_conversion(new_df, float(self.input_form.pixel_rate.text()))

            # Get inactive merged keys from excluded_rows (already computed earlier)
            excluded_keys = set(
                f"{row['image_name']}_{row['label']}"
                for df in excluded_rows
                for _, row in df.iterrows()
            )

            # Filter them out of new_df before adding to combined
            new_df = new_df[
                ~new_df.apply(lambda r: f"{r['image_name']}_{r['label']}" in excluded_keys, axis=1)
            ]

            combined_df = pd.concat([correct_df, new_df], ignore_index=True)
          
        else:
            combined_df = correct_df.copy()

        # --- Step 6: Construct excluded_df for inactive masks ---
        excluded_df = all_props_df[
            (all_props_df.apply(lambda r: f"{r['image_name']}_{r['label']}" in inactive_keys, axis=1)) &
            (~all_props_df.apply(lambda r: f"{r['image_name']}_{r['label']}" in merged_keys, axis=1))
        ].copy()

        # Add inactive merged masks from excluded_rows
        if excluded_rows:
            excluded_df = pd.concat([excluded_df, pd.concat(excluded_rows, ignore_index=True)], ignore_index=True)

        # --- Step 7: Normalize merged labels for consistency ---
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

        merged_individual_keys = set(group_map.keys())

        # Drop rows from combined_df if they correspond to any individual label that was merged
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

        #--- Step 8: Final deduplication ---
        combined_df = combined_df.drop_duplicates(subset=['image_name', 'label'])
        if not excluded_df.empty:
            excluded_df = excluded_df.drop_duplicates(subset=['image_name', 'label'])

        return combined_df, excluded_df

    def add_roi_name_column(self, df, is_merged=False, is_disconnected=False):
        def generate_name(row):
            base = os.path.splitext(str(row['image_name']))[0]
            label = str(row['label'])
            if is_merged:
                return f"{base}_label_{label}_merged.roi"
            elif is_disconnected:
                return f"{base}_label_{label}_disconnected.roi"
            elif label.startswith("drawn_"):
                return f"{base}_label_{label}_drawn.roi"
            else:
                return f"{base}_label{label}.roi"

        df['roi_name'] = df.apply(
            lambda r: generate_name(r)
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
        if hasattr(self, "gray_viewers"):
            for viewer in self.gray_viewers:
                viewer.connected_groups.clear()
                if hasattr(viewer, "mask_id_to_group"):
                    viewer.mask_id_to_group.clear()
                if hasattr(viewer, "new_mask_dict"):
                    viewer.new_mask_dict.clear()
            self.gray_viewers.clear()  # Remove old viewers

        # Clear image layout in GUI
        if self.image_layout is not None:
            while self.image_layout.count():
                child =  self.image_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

        self.image_layout = None  # Will be recreated during `create_images_tab()`

        # Reset worker and masks
        if hasattr(self, "worker"):
            self.worker.masks_dict.clear()
            self.worker.image_dict.clear()
            del self.worker 
        try:
            folder_path = self.input_form.images_folder_path.text()
            csv_file_name = self.input_form.csv_file_name.text()
            roi_folder_name = self.input_form.roi_folder_name.text()
            condition_name = self.input_form.condition_name.text()
            rep_num = self.input_form.rep_num.text()
            main_marker_identifier = self.input_form.unique_main_marker_identifier.text()
            nucleus_identifier = self.input_form.unique_nucleus_identifier.text()
            color = self.input_form.main_marker_channel_dropdown.currentText()
            # Get values from sliders instead of text fields
            main_marker_contrast_low = self.input_form.main_marker_low_contrast_slider.value()
            main_marker_contrast_high = self.input_form.main_marker_high_contrast_slider.value()
            nucleus_contrast_low = self.input_form.nucleus_low_contrast_slider.value()
            nucleus_contrast_high = self.input_form.nucleus_high_contrast_slider.value()
            diam = int(self.input_form.diameter.text())
            flow_thresh = float(self.input_form.flow_threshold.text())
            min_area = int(self.input_form.min_area.text())
            min_non_black_pixels_percentage = float(self.input_form.min_non_black_pixels_percentage.text())
            intensity_threshold = int(self.input_form.intensity_threshold.text())
            min_nucleus_pixels_percentage = float(self.input_form.min_nucleus_pixels_percentage.text())
            nucleus_pixel_threshold = int(self.input_form.nucleus_pixel_threshold.text())
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
        self.input_form.processing_label.setText(message)
        
    def show_save_all(self):
        self.image_layout.addWidget(self.button_widget)

    def processing_done(self):
        """
        Called when processing is finished
        """
        self.processing_in_progress = False
        self.input_form.process_button.setEnabled(True)
     
    def update_progress(self, value):
        """
        Update progress bar when processing is updated
        """
        self.input_form.progress_bar.setValue(value)  # Updating bar with percentage
        self.input_form.progress_bar.setFormat(f"{value}%")  # Display percentage
        self.input_form.progress_bar.repaint()  # Force UI update

    def add_images_to_scrollable_area(self, title, pixmap_gray, pixmap_rgb, pixmap_overlay, masks_list):

        container = QWidget()
        layout = QHBoxLayout()
        label_gray = QLabel()
        label_rgb = QLabel()
        label_overlay = QLabel()
        label_rgb.setScaledContents(True)
        label_overlay.setScaledContents(True)
        label_gray.setScaledContents(True)
        scaled_width = 430 # Target width for each image
        scaled_height = 530  # Target height for each image
        
        # Scale all images consistently
        scaled_pixmap_gray = pixmap_gray.scaled(scaled_width, scaled_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        scaled_pixmap_rgb = pixmap_rgb.scaled(scaled_width, scaled_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        scaled_pixmap_overlay = pixmap_overlay.scaled(scaled_width, scaled_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        label_rgb.setPixmap(scaled_pixmap_rgb)
        label_overlay.setPixmap(scaled_pixmap_overlay)
        label_gray.setPixmap(scaled_pixmap_gray)

        show_labels = self.input_form.cell_labels_checkbox.isChecked()
        font_size = self.input_form.cell_label_font_size_spinbox.value()
        # Using the scaled gray image in ImageViewer
        if not hasattr(self, "viewer_mode_controller"):
            self.viewer_mode_controller = ViewerModeController()
        controller = self.viewer_mode_controller    
        self.gray_viewer = ImageViewer(scaled_pixmap_gray, masks_list, font_size, show_labels, title, mode_controller=controller)

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

    def closeEvent(self, event):
        """
        Override the closeEvent to close the help window when the main window is closed
        """
        if self.help_text.isVisible():
            self.help_text.setVisible(False)  # Close help window before closing main window
        event.accept()  # Proceed with closing the main window


class ImageProcessingWorker(QThread):
    
    image_processed = pyqtSignal(str, QPixmap, QPixmap, QPixmap, list)
    status_update = pyqtSignal(str)  # New signal for status updates
    show_save_all = pyqtSignal() # Signal to show the save all button
    finished_processing = pyqtSignal() # Signal for the end of the process
    progress_updated = pyqtSignal(int) # Signal for a progress bar updates
    

    def __init__(self, images, folder_path, condition_name, rep_num, main_marker_identifier, nucleus_identifier,  color,  main_marker_contrast_low,
                 main_marker_contrast_high, nucleus_contrast_low, nucleus_contrast_high,  diam, thresh, min_area, min_non_black_pixels_percentage,
                 intensity_threshold, min_nucleus_pixels_percentage, nucleus_pixel_threshold, pixel_conv_rate, csv_file_name, roi_folder_name, progress_bar, model,
                 nucleus_channel_present=True):
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
        self.count = 0  # Initialize count for processed images
        self.progress_bar = progress_bar
        self.model = model
        self.nucleus_channel_present = nucleus_channel_present
        self.image_dict = {}
        self.masks_dict = {}

    def stop(self):
        self.active = False
        
    def run(self):
        
        main_marker_channel_value = self.color
        
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
        
        # Warning the user if they put in a path for an empty folder
        total_images = sum(1 for key in self.images if self.main_marker_identifier in key)
        if total_images == 0:
            self.status_update.emit("No images found in the folder. Check file IDs")
            self.finished_processing.emit() 
            return
        
        fail = True
        self.status_update.emit("Processing started...")
        self.all_props_df = pd.DataFrame()
        num_images = len(self.images)  # Getting total number of images
        self.masks_dict = {}  # Dictionary to store masks

        for num, (name, image) in enumerate(self.images.items()):
            if not self.active:
                break

            if self.main_marker_identifier in name and self.main_marker_identifier!="" and self.active:
                main_marker_image_name = name
                main_marker_image_path = image
                if not hasattr(self, "image_shape"):
                    image_array = np.array (Image.open(main_marker_image_path))
                    self.image_shape = image_array.shape
                if self.nucleus_channel_present:
                    nucleus_name = name.replace(self.main_marker_identifier, self.nucleus_identifier)
                    if (nucleus_name not in self.images or self.nucleus_identifier=="") and (self.active):  # Prevent KeyError
                        self.status_update.emit(f"Missing nucleus image: {nucleus_name}")
                        self.finished_processing.emit()
                        return
                    
                    nucleus_image_path = self.images[nucleus_name]
 
                if self.active:
                    try:
                        # Processing images
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
                        predicted_masks, _, _ = self.model.eval(main_marker_image, diameter=diamet, flow_threshold = self.thresh,  channels=[0, marker_channel_color])

                        if not self.active:
                            break
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

                        for mask, label in zip(masks_list, df['label']):
                            mask_key = f"{main_marker_image_name}{label}"
                            self.masks_dict[mask_key] = mask

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
                        print(e)
                        continue  # Moving to the next image

            self.progress_updated.emit(int(((num + 1) / num_images) * 100))
          
        if (fail and self.active and self.count >= 1) or (not self.active):
            self.all_props_df = pixel_conversion(self.all_props_df, self.pixel_conv_rate)
            self.status_update.emit(f"Processing completed! {self.count} images processed.")
            self.show_save_all.emit()
            self.finished_processing.emit() 
        
        elif fail:
            self.status_update.emit("No images processed. Please check your input parameters.")
            self.finished_processing.emit() 
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    main_window = ImageProcessingApp()
    main_window.show()
    sys.exit(app.exec())