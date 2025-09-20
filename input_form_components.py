from PyQt6.QtWidgets import QLabel, QPushButton, QFormLayout, QSpinBox
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtWidgets import QPushButton, QFileDialog, QComboBox, QSizePolicy
from PyQt6.QtWidgets import QLineEdit, QScrollArea, QComboBox, QTextEdit
from PyQt6.QtWidgets import QGridLayout, QSizePolicy, QProgressBar, QSlider, QCheckBox
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from cellpose import core


class ModelSelectorWidget(QWidget):
    """
    A custom QWidget that lets users choose between built-in or custom Cellpose models using a dropdown menu and optional file input.
    """
    # define a custom signal that will emit the model type and (optionally) custom model path
    model_changed = pyqtSignal(str, str)
    def __init__(self, font):
        super().__init__()
        self.model_type = "cyto3"  # default pre-trained cellpose model
        self.custom_model_path = None 

        # the dropdown menu for the model type
        self.setMinimumHeight(25)  
        self.setMaximumHeight(40)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["cyto3", "nuclei", "custom model"])
        self.model_dropdown.setCurrentText("cyto3")
        self.model_dropdown.setFont(font)
        self.model_dropdown.setMinimumHeight(25)
        self.model_dropdown.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.model_dropdown.setStyleSheet("""
            QComboBox {
                border: 1px solid gray;
                border-radius: 1px;
                padding: 1px 20px 1px 3px;  /* Adjust padding to make room for wider arrow */
                min-width: 6em;
                background-color: white;
                selection-color:black;
                selection-background-color: lightblue;
            }
            QComboBox:!editable, QComboBox::drop-down:editable {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #FFFFFF, stop: 0.4 #FFFFFF,
                                            stop: 0.5 ##FFFFFF, stop: 1.0 #FFFFFF);
            }
            /* When popup is open */
            QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #FFFFFF, stop: 0.4 #FFFFFF,
                                            stop: 0.5 #FFFFFF, stop: 1.0 #FFFFFF);
            }                                          
            /* Arrow inside dropdown */
            QComboBox::down-arrow {
                image: url(:/black_arrow.png);  /* Path to your arrow icon */
                width: 20px;  /* Set arrow width to 20px */
                height: 20px;  /* Optionally adjust arrow height */
            }
        """)
        self.model_dropdown.currentTextChanged.connect(self.on_model_selection_changed)
        dropdown_view = self.model_dropdown.view()
        dropdown_view.setMinimumWidth(50)
        dropdown_view.setMinimumHeight(30)
        dropdown_view.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        # the input components for the custom model 
        self.custom_model_input = QLineEdit()
        self.custom_model_input.setPlaceholderText("Path to the custom cellpose model")
        self.custom_model_input.setFont(font)
        self.custom_model_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #808080;
                border-radius: 3px;
                padding: 3px;
                background-color: white;   /* match the dropdown */
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2;  /* highlight color when active */
            }
        """)
        self.browse_button = QPushButton("Browse")
        self.browse_button.setFont(font)
        self.browse_button.setFixedSize(70, 25)
        self.browse_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;   /* light gray background */
                border: 1px solid gray;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #d9d9d9;   /* slightly darker on hover */
            }
            QPushButton:pressed {
                background-color: #bfbfbf;   /* darker when clicked */
            }
        """)
        # Connect dropdown change to handler function
        self.browse_button.clicked.connect(self.browse_custom_model)

        # The custom model section layout
        self.custom_model_container = QWidget()
        custom_layout = QHBoxLayout(self.custom_model_container)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        custom_layout.setSpacing(5)
        custom_layout.addWidget(self.custom_model_input)
        custom_layout.addWidget(self.browse_button)
        self.custom_model_container.setVisible(False)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self.model_dropdown)
        layout.addWidget(self.custom_model_container)
        layout.setStretch(0, 1)  # dropdown gets 1/4 width
        layout.setStretch(1, 3)  # input section gets 3/4 width
        self.setLayout(layout)
        self.model_dropdown.currentTextChanged.connect(self.on_model_selection_changed)


    def on_model_selection_changed(self, selected_text):
        """
        Handles changes in the model selection dropdown.

        If the user selects "custom model", show the input field and browse button 
        to allow selection of a custom model file. Otherwise, hide and disable them.
        """
        # Checking if the user selected "custom model"
        is_custom = selected_text == "custom model"  
        # Showing/hiding the input field and enabling/disabling the browse button
        self.custom_model_container.setVisible(is_custom)
        self.custom_model_input.setEnabled(is_custom)
        self.browse_button.setEnabled(is_custom)

        # Clearing any previously entered path if switching away from "custom model"
        if not is_custom:
            self.custom_model_input.setText("")
        # Notify other parts of the application about the model change
        self.emit_model_change()


    def browse_custom_model(self):
        """
        Opens a file dialog to let the user select a custom model file.
        Once a file is selected, updates the input field with the path 
        and emits the model change signal.
        """
        # Creating and configuring the file dialog
        file_dialog = QFileDialog(self, "Select Custom Model")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        # Showing the dialog and checking if the user selected a file
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                # Updating the text input with the selected file path
                self.custom_model_input.setText(selected_files[0])
                # Notifying the application that the model path has changed
                self.emit_model_change() 

    def emit_model_change(self):
        """
        Emits the "model_changed" signal with the selected model type and custom path.
        If a custom model is selected, the "model_type" is set to an empty string 
        and the "custom_path" contains the user-provided path. Otherwise, "model_type"
        holds the selected built-in model, and "custom_path" is empty.
        """
        if self.model_dropdown.currentText() == "custom model":
            model_type = "" # Indicates that a custom model is used
            custom_path = self.custom_model_input.text()
        else:
            model_type = self.model_dropdown.currentText()
            custom_path = ""  # No custom path needed for built-in models
        # Storing the current selection in the instance attributes
        self.model_type = model_type
        self.custom_model_path = custom_path
        # Emitting a signal to notify other components of the change
        self.model_changed.emit(model_type or "", custom_path or "")


class DraggableTextEdit(QTextEdit):
    """
    A custom QTextEdit widget that can be clicked and dragged around within its parent widget (for the help window)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_active = False # Flag to track whether dragging is active
        self._drag_position = QPoint() # Storing offset from top-left corner to mouse click position

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = True # Start dragging
            # Calculate offset between mouse click position and widget top-left corner
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()  # Accept the event to override default behavior
        else:
            super().mousePressEvent(event)  # Call the default behavior for other mouse buttons

    def mouseMoveEvent(self, event):
        if self._drag_active and event.buttons() & Qt.MouseButton.LeftButton:
            # Calculate the new top-left position of the widget during drag
            new_pos = event.globalPosition().toPoint() - self._drag_position

            if self.parent(): # If there is a parent widget, enforce boundaries
                parent_rect = self.parent().rect() # Get parent widget's geometry
                mapped_pos = self.parent().mapFromGlobal(new_pos) # Convert position to parent coordinates

                help_width = self.width() # Width of the QTextEdit
                help_height = self.height() # Height of the QTextEdit

                # Define the maximum and minimum allowed positions to keep widget within the parent
                max_x = max(parent_rect.width() - 400, 0) # Custom right-side margin
                max_y = max(parent_rect.height() - 200, 0) # Custom bottom-side margin
                min_x = -help_width + 50 # Custom left-side margin
                min_y = -help_height + 50 # Custom top-side margin

                # Clamp the new position within the allowed bounds
                x = max(min_x, min(mapped_pos.x(), max_x))
                y = max(min_y, min(mapped_pos.y(), max_y))

                # Convert back to global coordinates before moving
                new_pos = self.parent().mapToGlobal(QPoint(x, y))

            self.move(new_pos) # Move the widget to the new position
            event.accept()
        else:
            # Default behavior if not dragging
            super().mouseMoveEvent(event)
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = False # Stop dragging
            event.accept()
        else:
            # Call default behavior for other mouse buttons
            super().mouseReleaseEvent(event)

    def center_in_parent(self):
        """
        Centers the QTextEdit in its parent widget.
        """
        if not self.parent():
            return
        # Get parent and self dimensions
        parent_width = self.parent().width()
        parent_height = self.parent().height()
        help_width = self.width()
        help_height = self.height()
        # Calculate coordinates to center the widget
        help_x = (parent_width - help_width) // 2
        help_y = (parent_height - help_height) // 2
        self.move(help_x, help_y) # Move to the centered position


class InputFormWidget(QWidget):
    """
    A widget class that creates a structured, scrollable form interface for configuring 
    image segmentation and processing parameters.
    """
    processClicked = pyqtSignal(object)
    saveClicked = pyqtSignal()
    def __init__(self, parent=None, help_text=None, tab_widget=None):
        """
        Initializes the form widget.

        Args:
            parent (QWidget): Parent widget.
            help_text (QLabel): A label used for displaying help instructions.
            tab_widget (QWidget): Optional parent tab widget to integrate into.
        """
        super().__init__(parent)
        self.tabs = tab_widget
        self.create_input_form()
        self.help_text = help_text
        # Populate the help text QLabel
        self.help_text.setText(self.get_help_text())

    def create_slider(self, default_value, font_input):
        """
        Creates a horizontal slider paired with a label displaying its current value.
        """
        label = QLabel(f"{default_value}")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        slider.setValue(default_value)
        # Update the label dynamically when the slider moves
        slider.valueChanged.connect(lambda value: label.setText(f"{value}"))
        # Layout: horizontal container with spacing & margins to hold the slider and its value
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(0, 0, 15, 0)
        slider_layout.setSpacing(5)
        slider_layout.addWidget(slider)  # Adding the slider
        slider_layout.addWidget(label, 0, Qt.AlignmentFlag.AlignLeft)  # Adding the label to the right
        # Setting font and styling for the label and slider
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        label.setFont(font_input) 
        label.setMinimumWidth(20)
        label.setStyleSheet("""
            QLabel {
                font-size: 9px;
                color: black;
                padding-left: 15px;  # Adding a little space between slider and label
            }
        """)
        slider.setMinimumWidth(80) 
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Container for the layout
        container = QWidget()
        container.setLayout(slider_layout)
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return container, slider


    def toggle_help(self):
        """
        Toggles visibility of the help text widget.

        When made visible, the widget is resized and centered within the parent,
        with minimum and maximum dimensions enforced for readability.
        """
        if self.help_text.isVisible():
            self.help_text.hide()
        else:
            max_width = max(300, self.width() - 20)
            max_height = max(150, self.height() - 20)
            self.help_text.resize(min(800, max_width), min(500, max_height))
            self.help_text.setMinimumSize(300, 150)
            self.help_text.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.help_text.show()
            self.help_text.center_in_parent()
    

    def create_input_form(self):
        """
        Builds the main UI: a scrollable form populated with input fields, sliders, dropdowns,
        checkboxes, and the "Process Images" button with progress bar.
        """
        # Main vertical layout with scroll area
        main_layout = QVBoxLayout()  # Layout for aligning widgets to the left
        self.setLayout(main_layout)
        scroll_field = QScrollArea()
        scroll_field.setWidgetResizable(True) 
        scroll_field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Container widget and its internal layout
        input_container = QWidget()
        input_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 5, 15, 5)
        layout.setSpacing(5)  # controls spacing between each input row
        input_container.setLayout(layout)
        rows_layout = QVBoxLayout()
        rows_layout.setSpacing(10)

        # Grid layout for structuring input sections and their corresponding labels
        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(5, 5, 0, 0)
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(5)
        self.font_label = QFont("Arial", 18, QFont.Weight.Bold) # Font for the section labels
        self.font_input = QFont("Arial", 18) # Font for the input sections
        grid_layout.setColumnMinimumWidth(0, 100)  # Adjusting the minimum width for the first column (labels) 
        grid_layout.setColumnStretch(0, 0)  # label column: no stretch
        grid_layout.setColumnStretch(1, 1)  # input column: stretch to fill

        # Customizing the help button 
        self.help_button = QPushButton("?")
        self.help_button.setMinimumSize(30, 30)
        self.help_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.help_button.setFont(self.font_label)
        self.help_button.clicked.connect(self.toggle_help) # Connecting the button to a function that toggles help text visibility

        # Creating a widget for a help button
        top_right_container = QWidget()
        top_right_layout = QVBoxLayout(top_right_container)
        top_right_layout.setContentsMargins(0, 0, 0, 0)  
        top_right_layout.addStretch()  # Pushung button to the right
        top_right_layout.addWidget(self.help_button)
        layout.addWidget(top_right_container)
        layout.setAlignment(top_right_container, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)

        # Input fields with sliders for contrast thresholding
        self.main_marker_low_contrast_widget, self.main_marker_low_contrast_slider = self.create_slider(15, self.font_input)
        self.main_marker_high_contrast_widget, self.main_marker_high_contrast_slider = self.create_slider(99, self.font_input)
        self.nucleus_low_contrast_widget, self.nucleus_low_contrast_slider = self.create_slider(15, self.font_input)
        self.nucleus_high_contrast_widget, self.nucleus_high_contrast_slider = self.create_slider(99, self.font_input)

        # creating a widget for model selection
        self.model_selector_widget = ModelSelectorWidget(self.font_input)

        # Input Fields
        # Folder path input section styling
        self.images_folder_path = QLineEdit("")
        self.images_folder_path.setFont(self.font_input)
        self.images_folder_path.setMinimumHeight(30)   # match button height
        self.images_folder_path.setMaximumHeight(40)
        self.images_folder_path.setStyleSheet("""
            QLineEdit {
                border: 1px solid gray;
                border-radius: 3px;
                padding: 3px;   /* slightly more padding so text isn’t cramped */
            }
        """)
        # Folder path browse button styling
        self.folder_browse_button = QPushButton("Browse")
        self.folder_browse_button.setFont(self.font_input)
        self.folder_browse_button.setFixedSize(70, 25)
        self.folder_browse_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;   /* light gray background */
                border: 1px solid gray;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #d9d9d9;   /* slightly darker on hover */
            }
            QPushButton:pressed {
                background-color: #bfbfbf;   /* darker when clicked */
            }
        """)
        self.folder_browse_button.clicked.connect(self.browse_custom_path)
        self.folder_input_container = QWidget()
        folder_input_layout = QHBoxLayout(self.folder_input_container)
        folder_input_layout.setContentsMargins(0, 0, 0, 0)
        folder_input_layout.setSpacing(5)
        folder_input_layout.addWidget(self.images_folder_path, stretch=1)
        folder_input_layout.addWidget(self.folder_browse_button)

        # the rest of the input sections
        self.csv_file_name = QLineEdit("single_cell_morphology")
        self.roi_folder_name = QLineEdit("ROIs")
        self.condition_name = QLineEdit("")
        self.rep_num = QLineEdit("")
        self.unique_main_marker_identifier = QLineEdit("")
        self.unique_nucleus_identifier = QLineEdit("")
        self.diameter = QLineEdit("15")
        self.flow_threshold = QLineEdit("0.4")
        self.min_area = QLineEdit("150")
        self.min_non_black_pixels_percentage = QLineEdit("10")
        self.intensity_threshold = QLineEdit("70")
        self.min_nucleus_pixels_percentage = QLineEdit("10")
        self.nucleus_pixel_threshold = QLineEdit("200")
        self.pixel_rate = QLineEdit("") 

        # Setting parameters for input fields
        for input_field in [
            self.images_folder_path, self.csv_file_name, self.roi_folder_name, self.condition_name, self.rep_num, 
            self.unique_main_marker_identifier, self.unique_nucleus_identifier, self.diameter, self.flow_threshold,
            self.min_area, self.min_non_black_pixels_percentage, self.intensity_threshold,
            self.min_nucleus_pixels_percentage, self.nucleus_pixel_threshold, self.pixel_rate,
            self.main_marker_low_contrast_widget, self.main_marker_high_contrast_widget,
            self.nucleus_low_contrast_widget, self.nucleus_high_contrast_widget,
        ]:
            input_field.setFont(self.font_input)
            input_field.setMinimumWidth(100)
            input_field.setMinimumHeight(25)
            input_field.setMaximumHeight(40)
            input_field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            input_field.setStyleSheet("""
            QLineEdit {
                border: 1px solid gray; 
                border-radius: 3px;
                padding: 1px;
            }
        """)

        # Customizing dropdown menu for the segmentation marker color
        self.main_marker_channel_dropdown = QComboBox()
        self.main_marker_channel_dropdown.addItem("")  # Empty item as placeholder
        self.main_marker_channel_dropdown.addItems(["red", "green"])
        self.main_marker_channel_dropdown.setFont(self.font_input)
        self.main_marker_channel_dropdown.setMinimumWidth(100)  
        self.main_marker_channel_dropdown.setMinimumHeight(25)  
        self.main_marker_channel_dropdown.setMaximumHeight(40)
        self.main_marker_channel_dropdown.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.main_marker_channel_dropdown.setStyleSheet("""
            QComboBox {
                border: 1px solid gray;
                border-radius: 1px;
                padding: 1px 20px 1px 3px;  /* Adjust padding to make room for wider arrow */
                min-width: 6em;
                background-color: white;
                selection-color:black;
                selection-background-color: lightblue;
            }
            QComboBox:!editable, QComboBox::drop-down:editable {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #FFFFFF, stop: 0.4 #FFFFFF,
                                            stop: 0.5 ##FFFFFF, stop: 1.0 #FFFFFF);
            }
            /* When popup is open */
            QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #FFFFFF, stop: 0.4 #FFFFFF,
                                            stop: 0.5 #FFFFFF, stop: 1.0 #FFFFFF);
            }                                          
            /* Arrow inside dropdown */
            QComboBox::down-arrow {
                image: url(:/black_arrow.png);  /* Path to your arrow icon */
                width: 20px;  /* Set arrow width to 20px */
                height: 20px;  /* Optionally adjust arrow height */
            }
        """)
    
        # Creating a view for the dropdown menu
        dropdown_view = self.main_marker_channel_dropdown.view()
        dropdown_view.setMinimumWidth(100)
        dropdown_view.setMinimumHeight(30)
        dropdown_view.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.input_row_count = 1
        self.form_layout = QFormLayout()
        self.form_layout.setSpacing(10)
        self.form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.form_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # GPU checkbox
        GPU_checkbox_label = QLabel(f"{self.input_row_count}. GPU resources available")
        GPU_checkbox_label.setFont(self.font_label)
        # Create checkbox and set its state based on GPU availability
        self.GPU_checkbox = QCheckBox()
        self.GPU_checkbox.setStyleSheet("QCheckBox::indicator { width: 25px; height: 25px; }")
        # Check for GPU availability
        gpu_available = core.use_gpu()
        self.GPU_checkbox.setChecked(gpu_available)

        # Add first 6 rows in order
        self.add_row("Images Folder Path:", self.folder_input_container)
        self.add_row("Output File Name:", self.csv_file_name)
        self.add_row("ROI Folder Name:", self.roi_folder_name)
        self.add_row("Cellpose Model:", self.model_selector_widget)
        self.add_row("GPU Resources Available:", self.GPU_checkbox)
        self.add_row("Condition Name:", self.condition_name)
        self.add_row("Replicate #:", self.rep_num)
        self.add_row("Segmentation Channel File ID:", self.unique_main_marker_identifier)

        # Cell Labels Checkbox
        checkbox_cell_labels = QLabel(f"{self.input_row_count}. Display cell labels")
        checkbox_cell_labels.setFont(self.font_label)
        self.cell_labels_checkbox = QCheckBox()
        self.cell_labels_checkbox.setStyleSheet("QCheckBox::indicator { width: 25px; height: 25px; }")
        
        self.cell_label_font_size_spinbox = QSpinBox()
        self.cell_label_font_size_spinbox.setRange(5, 30)
    
        self.cell_label_font_size_spinbox.setValue(18)  # default value
        self.cell_label_font_size_spinbox.setSuffix(" pt")
        self.cell_label_font_size_spinbox.setVisible(False)  # hidden by default
        font_spinbox = QFont("Arial", 18)

        self.cell_label_font_size_spinbox.setFont(font_spinbox)
        self.cell_label_font_size_spinbox.setMinimumHeight(25) 
        self.cell_label_font_size_spinbox.setMaximumHeight(30)   
        

        # Horizontal layout to place checkbox + font size spinbox in one row
        cell_label_layout = QHBoxLayout()
        cell_label_layout.addWidget(self.cell_labels_checkbox)
        cell_label_layout.addWidget(self.cell_label_font_size_spinbox)
        # cell_label_layout.setSpacing(10)
        cell_label_layout.setContentsMargins(0, 0, 0, 0)
        cell_label_widget = QWidget()
        cell_label_widget.setLayout(cell_label_layout)
        # Add to the form layout
        self.form_layout.addRow(checkbox_cell_labels, cell_label_widget)

        # Toggle font size spinbox when checkbox is checked
        self.cell_labels_checkbox.toggled.connect(self.cell_label_font_size_spinbox.setVisible)
        self.input_row_count += 1
                
        # Nucleus checkbox
        checkbox_label = QLabel(f"{self.input_row_count}. Nucleus channel present")
        checkbox_label.setFont(self.font_label)
        self.nucleus_checkbox = QCheckBox()
        self.nucleus_checkbox.setStyleSheet("QCheckBox::indicator { width: 25px; height: 25px;}")
        nucleus_checkbox_layout = QHBoxLayout()
        nucleus_checkbox_layout.addWidget(self.nucleus_checkbox)
        nucleus_checkbox_layout.setContentsMargins(0, 0, 0, 0)
        nucleus_checkbox_layout.setSpacing(10)
        nucleus_checkbox_widget = QWidget()
        nucleus_checkbox_widget.setLayout(nucleus_checkbox_layout)
        self.form_layout.addRow(checkbox_label, nucleus_checkbox_widget)
        self.input_row_count += 1
        self.nucleus_group_widget = QWidget()
        self.nucleus_layout = QFormLayout(self.nucleus_group_widget)
        self.nucleus_layout.setSpacing(10)
        self.nucleus_layout.setContentsMargins(0, 0, 0, 0)
        self.nucleus_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    
        # Add all nucleus rows
        self.add_nucleus_row("  • Nucleus Channel File ID:", self.unique_nucleus_identifier)
        self.add_nucleus_row("  • Lower Percentile of Pixel Intensities for Nucleus Channel:", self.nucleus_low_contrast_widget)
        self.add_nucleus_row("  • Upper Percentile of Pixel Intensities for Nucleus Channel:", self.nucleus_high_contrast_widget)
        self.add_nucleus_row("  • Minimum Percentage of Cell Area Occupied by Nucleus:", self.min_nucleus_pixels_percentage)
        self.add_nucleus_row("  • Nucleus Channel Intensity Threshold:", self.nucleus_pixel_threshold)

        self.nucleus_group_widget.setVisible(False)
        self.nucleus_checkbox.toggled.connect(self.nucleus_group_widget.setVisible)
        self.form_layout.addRow(self.nucleus_group_widget)  # 👈 Add the whole block as a form row

        # Add remaining fields
        self.add_row("Segmentation Channel Colour:", self.main_marker_channel_dropdown)
        self.add_row("Lower Percentile of Pixel Intensities for Segmentation Marker Channel:", self.main_marker_low_contrast_widget)
        self.add_row("Upper Percentile of Pixel Intensities for Segmentation Marker Channel:", self.main_marker_high_contrast_widget)
        self.add_row("Average Cell Diameter (µm):", self.diameter)
        self.add_row("Flow Threshold:", self.flow_threshold)
        self.add_row("Minimum Cell Area (µm²):", self.min_area)
        self.add_row("Minimum Percentage of Image Occupied by Cells:", self.min_non_black_pixels_percentage)
        self.add_row("Segmentation Channel Intensity Threshold:", self.intensity_threshold)
        self.add_row("Pixel-to-Micron Ratio:", self.pixel_rate)

        # Attach form layout to the main layout
        layout.addLayout(self.form_layout)
                
        # Customizing the start button
        self.process_button = QPushButton("Process Images")
        self.process_button.setFont(QFont("Arial", 25))
        self.process_button.setMinimumSize(100, 40)
        self.process_button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        self.process_button.setStyleSheet("""
            QPushButton {
                font-size: 20pt;  /* Bigger font size */
                padding: 5px;    /* Add padding around the text */
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
        self.processing_label = QLabel("")
        self.processing_label.setStyleSheet(
            "color: green; font-size: 22px; font-weight: bold; padding-top: -20px;"
        )
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.process_button)
        button_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        button_layout.setContentsMargins(5, 15, 15, 0)
        layout.addLayout(button_layout)
    
        processing_layout = QHBoxLayout()
        processing_layout.addWidget(self.processing_label)
    
        self.processing_label.setMinimumHeight(40)
        self.processing_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        processing_layout.setContentsMargins(5, 15, 15, 0)
        layout.addLayout(processing_layout)  # Add layout to main layout
        layout.addStretch()

        # Progress bar 
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ADD8E6;  /* Light blue border */
                border-radius: 5px;
                background-color: white;  /* Background color */
                text-align: center;  /* Center align text */
                font-size: 18pt;  /* Larger text */
                color: black;  /* Black text */
                padding: 3px;
            }
            QProgressBar::chunk {
                background-color: lightblue; /* Light blue progress bar */
                width: 10px;
            }
        """)
        scroll_field.setWidget(input_container)
        main_layout.addWidget(scroll_field)
        self.process_button.clicked.connect(lambda: self.processClicked.emit(button_layout))
    

    def browse_custom_path(self):
        """
        Opens a file dialog to let the user select a folder with images.
        Once a file is selected, updates the input field with the path 
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.images_folder_path.setText(folder)
            

    def add_row(self, label_text, input_widget):
        """
        Add a new row to the main input form layout with a numbered label and an associated input widget.
        """
        numbered_label = f"{self.input_row_count}. {label_text}"
        self.input_row_count += 1

        label = QLabel(numbered_label)
        label.setFont(self.font_label)
        label.setMinimumHeight(25)
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        input_widget.setMinimumHeight(25)
        input_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.form_layout.addRow(label, input_widget)

    def add_nucleus_row(self, label_text, input_widget):
        """
        Add a new row to the nucleus-specific layout with a label and an input widget.
        """
        label = QLabel(label_text)
        label.setFont(self.font_label)
        label.setMinimumHeight(25)
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        input_widget.setFont(self.font_input)
        input_widget.setMinimumHeight(25)
        input_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.nucleus_layout.addRow(label, input_widget)


    def get_help_text(self):
        """
        Returns a multi-line rich-text help string with bolded parameter names.
        """
        return (
            "<b>1. Images Folder Path:</b> The path to the folder containing the images. Only single-channel images should be included. "
            "If multi-channel images with the same file IDs are present, you may see a “No images have been processed” message. Files without the .tif, .tiff, .TIF, or .TIFF extensions will be ignored if present in the same folder.<br><br>"

            "<b>2. Output File Name:</b> The desired name for the output .csv file. This is not a file path, just the name without the .csv extension. The file will be saved in the Images Folder.<br><br>"

            "<b>3. ROI Folder Name:</b> The desired name for the folder that will contain the ROIs of the selected cells. This is not a folder path, just the name. The folder will be created inside the Images Folder.<br><br>"

            "<b>4. Cellpose Model:</b> Select one of the available options or enter the path to your custom pre-trained model saved locally.<br><br>"

            "<b>5. GPU Resources Available:</b> If checked, the segmentation process will run using the GPU. Always enable this option on Mac (M1–M3).<br><br>"

            "<b>6. Condition Name:</b> Additional column with the specified condition that will be added to the .csv file.<br><br>"

            "<b>7. Replicate #:</b> Additional column with the replicate # specified that will be added to the .csv file.<br><br>"

            "<b>8. Segmentation Channel File ID:</b> A keyword unique to images containing a segmentation marker (e.g., d2, ch1).<br><br>"

            "<b>9. Display Cell Labels:</b> If checked, labels will be displayed on top of each segmented object using the specified font for the digits.<br><br>"

            "<b>10. Nucleus Channel Present:</b> If checked, additional input fields for specifying the nucleus channel input parameters will be displayed including:<br>"
            "&emsp;<b>- Nucleus Channel File ID:</b> A keyword unique to images containing a nuclear marker (e.g., d0, ch2).<br>"
            "&emsp;<b>- Lower Percentile of Pixel Intensities for Nucleus Channel:</b> Any intensity below this percentile is mapped to 0 (black). Contrast adjustments are for visualization only; fluorescence intensity is extracted from raw images.<br>"
            "&emsp;<b>- Upper Percentile of Pixel Intensities for Nucleus Channel:</b> Any intensity above this percentile is mapped to 1 (white).<br>"
            "&emsp;<b>- Minimum Percentage of Cell Area Occupied by Nucleus:</b> The minimum proportion of the cell's area that must be occupied by nucleus pixels.<br>"
            "&emsp;<b>- Nucleus Channel Intensity Threshold:</b> Minimum fluorescence intensity for a pixel to be considered part of the nucleus.<br><br>"

            "<b>11. Segmentation Channel Color:</b> The color of the segmentation channel for display (choose from dropdown).<br><br>"

            "<b>12. Lower Percentile of Pixel Intensities for Segmentation Marker Channel:</b> Any intensity below this percentile is mapped to 0 (black). Contrast adjustments are for visualization only.<br><br>"

            "<b>13. Upper Percentile of Pixel Intensities for Segmentation Marker Channel:</b> Any intensity above this percentile is mapped to 1 (white).<br><br>"

            "<b>14. Average Cell Diameter:</b> The typical cell diameter in microns.<br><br>"

            "<b>15. Flow Threshold:</b> Maximum allowed flow error per segmented mask. Increase if you're missing ROIs; decrease to reduce noise.<br><br>"

            "<b>16. Minimum Cell Area:</b> Minimum area (in microns²) for a segmented object to be considered a valid cell.<br><br>"

            "<b>17. Minimum Percentage of Image Occupied by Cells:</b> Minimum proportion of the image that must be covered by cells. Increase this to ignore empty images.<br><br>"

            "<b>18. Segmentation Channel Intensity Threshold:</b> Minimum fluorescence intensity required for a pixel to be included in segmentation. Increase if empty images are segmented.<br><br>"

            "<b>19. Pixel-to-Micron Ratio:</b> The conversion factor from pixels to microns (depends on your microscope setup).<br><br>"


            "<b>• Toggle:</b> This is the default mode. It allows the user to remove incorrect masks from the analysis. Removed masks are stored temporarily in memory and can be toggled back while the application is still running.<br>"
            "<b>• Connect:</b> This mode allows the user to connect or disconnect neighboring objects (unlimited number) using a mouse stroke. Masks can be connected and disconnected multiple times while the application is still running.<br>"
            "<b>• Draw:</b> This mode allows the user to draw masks for cells that were not segmented by Cellpose. The drawn shapes should be as enclosed as possible. Small gaps in the outlines will likely be closed automatically, but larger gaps may lead to incorrect measurements of morphological properties.<br>"
            "<b>• Erase:</b> This mode allows the user to erase the drawn outlines.<br>"
        )



   


