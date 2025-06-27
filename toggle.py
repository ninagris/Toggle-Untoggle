import numpy as np
import cv2
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt

class ClickableMask(QGraphicsPixmapItem):
    """
    Allows toggling of the mask opacity when it's clicked
    """
    def __init__(self, pixmap, name, label, click_callback):
        super().__init__(pixmap)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.active_opacity = 1.0
        self.inactive_opacity = 0.2
        self.setOpacity(self.active_opacity)  # Start fully visible
        self.label = label  # Store the label for this mask
        self.name = name # Store the name for this mask
        self.is_inactive = False  # Track if mask is dimmed
        self.click_callback = click_callback  # Store the callback function

    def mousePressEvent(self, event):
        """
        Toggle between full and dimmed opacity
        """
        if self.opacity() == self.active_opacity:
            self.setOpacity(self.inactive_opacity)  # Dim it
            self.is_inactive = True  # Set the state to inactive
        else:
            self.setOpacity(self.active_opacity)  # Restore full visibility
            self.is_inactive = False  # Set the state back to active

        # Call the callback function to handle the click
        self.click_callback(self.name, self.label, not self.is_inactive)

        event.accept()

class ImageViewer(QGraphicsView):
    def __init__(self, pixmap, masks, colors=None):
        super().__init__()

        self.callback_dict = {} # Collect callback functions for each mask

        # Disable scroll bars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setStyleSheet("border: none; padding: 0px; margin: 0px;")

        # Set up a container for the image
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setFixedSize(pixmap.size())
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)

        self.mask_items = [] # For storing mask items added to the scene

        # Generating a unique color for each mask
        num_masks = len(masks)
        colors = self.generate_colors(num_masks)
        self.set_togglable_masks(masks, colors, pixmap)

    def set_togglable_masks(self, masks, colors, pixmap):
        """
        Making each masks a togglable object with assigned properties
        """
        for i, mask_data in enumerate(masks):
            label = mask_data["label"]  # Extract label
            name = mask_data["image_name"] # Extract the name of the image that the mask has been derived from
            mask = mask_data["mask"]  # Extract mask
            color = colors[i] 
            mask_pixmap = self.convert_mask_to_pixmap(mask, color) 
            # Scale the mask to match the image size
            mask_pixmap = mask_pixmap.scaled(pixmap.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            # Make each mask clickable and put it on top of an image
            mask_item = ClickableMask(mask_pixmap, name, label, click_callback=self.mask_click_callback)
            mask_item.setZValue(1)  # Ensure masks are on top
            self.scene.addItem(mask_item)
            self.mask_items.append(mask_item)

    # override method
    def resizeEvent(self, event):
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)  # Keep it fixed
        event.accept()

    def generate_colors(self, num_colors):
        """
        Generate distinct colors using HSV space
        """
        colors = []
        for i in range(num_colors):
            hue = int((i * 360 / num_colors) % 360)  # Spread colors evenly across HSV
            qt_color = QColor.fromHsv(hue, 255, 255, 255)  # White-tinted for better contrast
            colors.append((qt_color.red(), qt_color.green(), qt_color.blue(), 200))  # Convert to RGBA
        return colors
    
    def convert_mask_to_pixmap(self, mask, color):
        """
        Convert a NumPy mask array to a QPixmap with transparency and a unique color
        """
        height, width = mask.shape
        # Create an RGBA image
        colored_mask = np.zeros((height, width, 4), dtype=np.uint8)
        colored_mask[..., 0] = color[0]  # Red
        colored_mask[..., 1] = color[1]  # Green
        colored_mask[..., 2] = color[2]  # Blue
        colored_mask[..., 3] = mask * color[3]  # Alpha transparency (only for mask pixels)

        q_image = QImage(colored_mask.data, width, height, width * 4, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(q_image)
    
    def mask_click_callback(self, name, label, is_active):
        """
        Update or add entry for the mask clicked
        """
        self.callback_dict[f"{name}{label}"] = {'name': name, 'label': label, 'is_active': is_active}  


