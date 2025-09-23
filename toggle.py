
import numpy as np

from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem, QGraphicsPathItem
from PyQt6.QtGui import QPixmap, QImage, QColor, QFont, QPen, QPainter, QPainterPath


class ViewerModeController:
    """
    Controller class to manage viewer modes (toggle, connect, draw, erase).
    Keeps all registered viewers in sync when the mode changes.
    """
    def __init__(self):
        self._mode = "toggle"  # Set a default mode
        self.viewers = [] # List to store all registered ImageViewer instances

    def register_viewer(self, viewer):
        """
        Register a new viewer and sync it with the current mode.
        Also enables mode checking for that viewer.
        """
        self.viewers.append(viewer)
        viewer.set_mode(self._mode)  # Set the current shared mode to the new viewer
        viewer.mode_check_enabled = True # Enable mode-dependent behavior in the viewer

    def set_mode(self, new_mode):
        """
        Change the mode and propagate it to all registered viewers.
        For example, switching between 'connect', 'draw', 'toggle', etc.
        """
        # Normalize input: if empty string, treat as None
        if new_mode == "":
            new_mode = None
        
        self._mode = new_mode
        for viewer in self.viewers:
            viewer.set_mode(new_mode) # Update each viewer's mode

    def get_mode(self):
        """
        Returns the current shared mode.
        """
        return self._mode
    
    def sync_all_viewers(self):
        """
        Forces all viewers to re-sync with the current mode.
        """
        for viewer in self.viewers:
            viewer.set_mode(self._mode)


class ClickableMask(QGraphicsPixmapItem):
    """
    Custom QGraphicsPixmapItem subclass for handling individual mask interactivity.
    Supports opacity toggling, tracking active/inactive state, and integration with viewer callbacks.
    """
    def __init__(self, pixmap, name, label, click_callback, binary_mask, connection_mode_getter, viewer):
        """
        Initialize the ClickableMask item.

        Parameters:
        - pixmap: QPixmap representing the mask.
        - name: Name of the image this mask belongs to.
        - label: Unique label for the mask within the image.
        - click_callback: Function to call when the mask is toggled.
        - binary_mask: Binary array representation of the mask.
        - connection_mode_getter: Callable that returns whether connection mode is active.
        - viewer: The parent viewer object managing this mask.
        """
        super().__init__(pixmap)
        self.setAcceptHoverEvents(True) # Enable hover events if needed later
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton) # Accept left mouse clicks
        self.active_opacity = 1.0
        self.inactive_opacity = 0.2
        self.setOpacity(self.active_opacity) # Start in active (toggled) state

        self.label = label           # Label associated with this mask
        self.name = name             # Name (usually image or group identifier)
        self.is_inactive = False     # Track if mask is dimmed (inactive)
        self.binary_mask = binary_mask
        self.connection_mode_getter = connection_mode_getter # Check if in "connect" mode
        self.click_callback = click_callback # Callback function for clicks
        self.viewer = viewer # Reference to the viewer managing masks

    def mousePressEvent(self, event):
        """
        Handles the mouse press interaction for toggling mask visibility.
        Only works if the viewer is in 'toggle' mode and not in a connection mode.
        """
        if not self.viewer.mode_check_enabled:
            event.ignore()
            return

        viewer_mode = self.viewer.mode
        connection_mode = self.connection_mode_getter()

        # Only allow interaction in toggle mode, not in connect or draw mode
        if viewer_mode != "toggle" or connection_mode:
            event.ignore()
            return
        
        self.handle_toggle_mode()
        event.accept()

    def handle_toggle_mode(self):
        """
        Toggle mask or group of masks depending on whether this mask is part of a group.
        """
        mask_id = self.viewer.get_mask_id(self.name, self.label)
        group = self.viewer.mask_id_to_group.get(mask_id)
        # If grouped with other masks, toggle them together
        if group:
            self.toggle_group(group)
        else:
            self.toggle_individual()

    def toggle_group(self, group):
        """
        Toggle visibility for a group of masks.
        """
        # Determine current state from any one mask (they should be synchronized)
        turning_off = self.opacity() == self.viewer.active_opacity

        # Create a key for the merged/grouped entry
        merged_key = self.viewer.generate_merged_key(self.name, group["mask_ids"])

        # Update merged entry in callback dict
        self.viewer.callback_dict[merged_key] = {
            "name": self.name,
            "label": merged_key.split('_', 1)[1],
            "is_active": not turning_off,
            "merged": True
        }
        # Update each mask in the group
        for mid in group["mask_ids"]:
            item = self.viewer.get_item_by_id(mid)
            if item:
                item.setOpacity(self.viewer.inactive_opacity if turning_off else self.viewer.active_opacity)
                item.is_inactive = turning_off
                individual_key = f"{item.name}_{item.label}"
                if individual_key in self.viewer.callback_dict:
                    self.viewer.callback_dict[individual_key]["is_active"] = not turning_off

        # Remove merged mask from new_mask_dict if turning off
        if turning_off and merged_key in self.viewer.new_mask_dict:
            del self.viewer.new_mask_dict[merged_key]

    def toggle_individual(self):
        """
        Toggle visibility of a single mask and notify viewer of the change.
        """
        turning_off = self.opacity() == self.viewer.active_opacity
        self.setOpacity(self.viewer.inactive_opacity if turning_off else self.viewer.active_opacity)
        self.is_inactive = turning_off
        # Notify the viewer that this individual mask was toggled
        self.viewer.mask_click_callback(self.name, self.label, not turning_off)


class ImageViewer(QGraphicsView):
    """
    Main class for displaying images and associated interactive masks.
    Supports toggling masks, connecting/disconnecting them, drawing overlays,
    and syncing with other viewers via ViewerModeController.
    """
    def __init__(self, pixmap, masks, font_size, show_labels=False, colors=None, image_name=None, mode_controller=None):
        """
        Initializes the ImageViewer widget.
        
        Args:
            pixmap (QPixmap): The base image to display.
            masks (list): List of dictionaries with keys: 'mask', 'label', 'image_name'.
            font_size (int): Size of label fonts.
            show_labels (bool): Whether to show label text on masks.
            colors (list): Optional list of RGBA tuples to color the masks.
            image_name (str): Name of the image, used as mask prefix.
            mode_controller: Optional ViewerModeController for syncing modes across viewers.
        """
        super().__init__()

        # Callback information for individual masks
        self.callback_dict = {} # Collect callback functions for each mask
        self.original_masks = masks
        self.mode_controller = mode_controller  
        self.mode = "toggle"

        # Register to sync with other viewers if mode controller is used
        if self.mode_controller:
            self.mode_controller.register_viewer(self)
        else:
            self.set_mode("toggle")

        # Flags and state variables
        self.connection_mode = self.mode == "connect"
        self.disconnect_mode = self.mode == "connect"
        self.mouse_path = []
        self.connection_line = None
        self.connected_groups = []  # list of mask group dicts
        self.mask_id_to_group = {}  # mask_id -> group dict
        self.disconnect_mode = False
        self.new_mask_dict = {}
        self.correction_action = "connect"  # or "disconnect", depending on your UI
        
        # Canvas for drawing
        self.drawing = self.mode == "draw"
        self.last_draw_point = None
        self.image_name = image_name
        self.drawing_canvas = QPixmap(pixmap.size())
        self.drawing_canvas.fill(Qt.GlobalColor.transparent)
        self.drawing_item = QGraphicsPixmapItem(self.drawing_canvas)
        self.drawing_item.setZValue(999)  # On top of masks

        # Display settings
        self.active_opacity = 1.0
        self.inactive_opacity = 0.2
        self.pre_merge_callback_state = {}
        self.set_view_properties()

        # Set up graphics scene
        self.graphics_scene = QGraphicsScene(self)
        self.setScene(self.graphics_scene)
        self.graphics_scene.addItem(self.drawing_item)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.graphics_scene.addItem(self.pixmap_item)

        self.mask_items = [] # For storing mask items added to the scene

        # Generating a unique color for each mask
        num_masks = len(masks)

        # Generate colors if not provided
        colors = self.generate_colors(num_masks)
        self.set_togglable_masks(masks, colors, pixmap, font_size=font_size, show_labels=show_labels)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # Zoom limits
        self.current_zoom = 1.0
        self.min_zoom = 1.0
        self.max_zoom = 5.0
        # Enable pinch gestures for the trackpad
        self.grabGesture(Qt.GestureType.PinchGesture)

    def wheelEvent(self, event): 
        """
        Handle mouse wheel events for zooming.
        - Ignores touchpad two-finger scrolling
        - Zooms in/out when using a physical mouse wheel
        """
        if not event.pixelDelta().isNull():
            # Ignore touchpad scrolling
            event.ignore()
            return
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor # Determine zoom direction based on wheel rotation
        self.apply_zoom(factor) # Apply the zoom

    def event(self, event):
        """
        Handle general events, including gestures.
        - Detects pinch gestures from a trackpad
        - Calls apply_zoom() with the pinch scale factor
        """
        if event.type() == QEvent.Type.Gesture:
            pinch = event.gesture(Qt.GestureType.PinchGesture)
            if pinch:
                # Only use the change in scale factor
                self.apply_zoom(pinch.scaleFactor())
                return True  # Mark gesture as handled

        # Fall back to default event handling
        return super().event(event)

    def apply_zoom(self, factor):
        """
        Apply zoom to the QGraphicsView, clamped by min/max zoom limits.

        """
        # Calculate new zoom and clamp it
        new_zoom = self.current_zoom * factor
        if new_zoom < self.min_zoom:
            factor = self.min_zoom / self.current_zoom
            self.current_zoom = self.min_zoom
        elif new_zoom > self.max_zoom:
            factor = self.max_zoom / self.current_zoom
            self.current_zoom = self.max_zoom
        else:
            self.current_zoom = new_zoom
        # Apply the scaling transformation
        self.scale(factor, factor)
    
    def set_view_properties(self):
        """Configures the view to remove scrollbars and borders, center the view, and disable drag."""
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setStyleSheet("border: none; padding: 0px; margin: 0px;")

    def generate_merged_key(self, name, mask_ids):
        labels = sorted(str(self.get_item_by_id(mid).label) for mid in mask_ids)
        return  f"{name}({','.join(labels)})"
   
    def keyPressEvent(self, event):
        """Handles key events to switch between modes."""
        if event.key() == Qt.Key.Key_C:
            self.mode_controller.set_mode("connect")
            self.correction_action = "connect"
        elif event.key() == Qt.Key.Key_D:
            self.mode_controller.set_mode("connect")
            self.correction_action = "disconnect"
        elif event.key() == Qt.Key.Key_T:
            self.mode_controller.set_mode("toggle")

    def is_connection_mode(self):
        """Returns whether the viewer is in 'connect' mode."""
        return self.connection_mode
    
    def mousePressEvent(self, event):
        """Handles initiating interactions depending on the current mode."""
        if self.mode == "connect":
            self.mouse_path = [self.mapToScene(event.position().toPoint())]
            event.accept()
        elif self.mode == "draw":
            self.drawing = True
            self.last_draw_point = self.mapToScene(event.position().toPoint())
            event.accept()
        elif self.mode == "erase":
            self.drawing = True
            self.last_draw_point = self.mapToScene(event.position().toPoint())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handles drawing, erasing, or connecting based on the current mode."""
        # ===== CONNECT MODE =====
        if self.mode == "connect" and event.buttons() & Qt.MouseButton.LeftButton:
            # Convert the current mouse position to scene coordinates
            scene_point = self.mapToScene(event.position().toPoint())
            self.mouse_path.append(scene_point)

             # Remove the existing temporary path (if any)
            if self.connection_line:
                self.graphics_scene.removeItem(self.connection_line)

            # Create a new path from recorded mouse points
            path = QPainterPath()
            path.moveTo(self.mouse_path[0])
            for pt in self.mouse_path[1:]:
                path.lineTo(pt)

            # Set pen style and create a QGraphicsPathItem for the path
            pen = QPen(QColor("white"))
            pen.setWidth(2)
            self.connection_line = QGraphicsPathItem(path)
            self.connection_line.setPen(pen)
            self.connection_line.setZValue(200)
            # Add the path to the scene
            self.graphics_scene.addItem(self.connection_line)

            event.accept()

        # ===== DRAW MODE =====
        elif self.mode == "draw" and self.drawing:
            # Convert current position to scene coordinates
            current_point = self.mapToScene(event.position().toPoint())

             # If we have a valid previous point, draw a red line from it
            if self.last_draw_point is not None:  
                painter = QPainter(self.drawing_canvas)
                pen = QPen(QColor("red"), 3, Qt.PenStyle.SolidLine)
                painter.setPen(pen)
                painter.drawLine(self.last_draw_point, current_point)
                painter.end()

                #  Update the QGraphicsPixmapItem to show the new drawing
                self.drawing_item.setPixmap(self.drawing_canvas)
            # Update the last point to current for next segment
            self.last_draw_point = current_point  
            event.accept()

        # ===== ERASE MODE =====
        elif self.mode == "erase" and self.drawing:
            current_point = self.mapToScene(event.position().toPoint())

            if self.last_draw_point is not None:
                painter = QPainter(self.drawing_canvas)
                 # Set composition mode to clear (erase) pixels
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)

                # Set a transparent pen to simulate erasing
                eraser_size = 30
                pen = QPen(QColor(0, 0, 0, 0), eraser_size)
                painter.setPen(pen)
                # Draw a line to erase from the last to current point
                painter.drawLine(self.last_draw_point, current_point)
                painter.end()
                # Update the displayed pixmap
                self.drawing_item.setPixmap(self.drawing_canvas)

            self.last_draw_point = current_point
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handles mouse release events depending on the mode."""
        # ===== CONNECT MODE =====
        if self.mode == "connect" and event.button() == Qt.MouseButton.LeftButton:
            # Add the final point to the path
            self.mouse_path.append(self.mapToScene(event.position().toPoint()))
            # Handle connection logic (e.g., connect selected masks)
            self.handle_mouse_path()
            # Clear path and remove temporary drawing
            self.mouse_path = []
            if self.connection_line:
                self.graphics_scene.removeItem(self.connection_line)
                self.connection_line = None

            event.accept()

         # ===== DRAW / ERASE MODES =====
        elif self.mode in ("draw", "erase") and event.button() == Qt.MouseButton.LeftButton:
            # Stop drawing/erasing and reset last point
            self.drawing = False
            self.last_draw_point = None
            event.accept()

        else:
            super().mouseReleaseEvent(event)

    def extract_color_from_pixmap(self, pixmap):
        """Extracts the first non-transparent pixel's color from a pixmap."""
        image = pixmap.toImage()
        width, height = image.width(), image.height()
        
        # Scan the image pixel by pixel
        for y in range(height):
            for x in range(width):
                pixel = image.pixelColor(x, y)
                if pixel.alpha() > 0: # Check if pixel is not fully transparent
                    return (pixel.red(), pixel.green(), pixel.blue(), pixel.alpha())
         # Return white if no visible pixel found
        return (255, 255, 255, 255) 
    
    def set_mode(self, mode):
        """Sets the viewer mode to one of: 'toggle', 'connect', 'draw', 'erase'."""
        valid_modes = ("toggle", "connect", "draw", "erase")
    
        if mode is None or mode == "":
            # No active mode: disable interactions or set to a default "inactive" state
            self.mode = None
            self.connection_mode = False
            self.disconnect_mode = False
            self.drawing = False
            self.last_draw_point = None
            return
        
        if mode not in valid_modes:
            print(f"Invalid mode: {mode}")
            return
        
        self.mode = mode
        self.connection_mode = (mode == "connect")
        self.disconnect_mode = (mode == "connect")

    def handle_mouse_path(self):
        """
        Process the completed mouse stroke for connection.
        Either merges selected masks or disconnects an existing group.
        """
        hit_masks = self.get_hit_masks_from_path()
        active_items = [m for m in hit_masks if not m.is_inactive]

        if len(active_items) <= 1:
            return # Not enough masks to connect
        
         # Create a merged mask preview
        merged_mask = self.create_merged_mask(active_items)
         # Get unique IDs for the selected masks
        mask_ids = {self.get_mask_id(m.name, m.label) for m in active_items}

        # If all hit masks are part of the same group, disconnect them
        if self.check_and_disconnect(hit_masks):
            return
         # Otherwise, merge and connect the selected masks
        self.merge_and_connect_masks(active_items, mask_ids, merged_mask)
    
    def get_hit_masks_from_path(self):
        """
        Returns a set of masks touched by the mouse stroke path.
        """
        hit_masks = set()
        for item in self.mask_items:
            if item.is_inactive:
                continue
            for pt in self.mouse_path:
                # Convert from scene to image coordinates
                scene_x, scene_y = pt.x(), pt.y()
                img_x = int(scene_x * item.binary_mask.shape[1] / self.pixmap_item.pixmap().width())
                img_y = int(scene_y * item.binary_mask.shape[0] / self.pixmap_item.pixmap().height())
                # Check if the point falls within the binary mask
                if 0 <= img_x < item.binary_mask.shape[1] and 0 <= img_y < item.binary_mask.shape[0]:
                    if item.binary_mask[img_y, img_x]:
                        hit_masks.add(item)
                        break  # Only need one hit point per mask
        return hit_masks
    
    def create_merged_mask(self, active_items):
        """
        Combines binary masks into one merged mask and generates metadata.
        """
        merged_mask = np.zeros_like(active_items[0].binary_mask, dtype=np.int32)
        label_names = []

        for item in active_items:
            merged_mask[item.binary_mask > 0] = 1
            label_names.append(str(item.label))

        label_names = sorted(label_names)
        combined_label = "_".join(label_names)
        name = active_items[0].name
        mask_key = f"{name}_({combined_label})"

        # Store the merged mask and update internal dictionaries
        self.new_mask_dict[mask_key] = {
            "mask": merged_mask,
            "source": "connect",
            "image_name": name,
            "label_group": label_names
        }

        self.callback_dict[mask_key] = {
            "name": name,
            "label": mask_key,
            "is_active": True,
            "merged": True
        }

        return merged_mask
    
    def check_and_disconnect(self, hit_masks):
        """Check if all hit masks are part of the same group and disconnect them if so."""
        connected_mask_ids = {
            self.get_mask_id(m.name, m.label)
            for m in hit_masks if self.get_mask_id(m.name, m.label) in self.mask_id_to_group
        }

        # All masks must belong to some group
        if len(connected_mask_ids) == len(hit_masks):
            group_ids = [self.mask_id_to_group[mid] for mid in connected_mask_ids]
            if len(set(map(id, group_ids))) == 1: # All in the same group
                group = self.mask_id_to_group.get(next(iter(connected_mask_ids)))
                if group:
                    self.disconnect_group(group)
                    return True
        return False
    
    def merge_and_connect_masks(self, active_items, mask_ids, merged_mask):
        """Merge selected masks into a group and recolor them with a unified color."""
        # Find and unify existing groups
        groups = [self.mask_id_to_group[mid] for mid in mask_ids if mid in self.mask_id_to_group]
        seen = set()
        unique_groups = []
        for group in groups:
            group_id = id(group)
            if group_id not in seen:
                seen.add(group_id)
                unique_groups.append(group)

        # Merge all mask IDs and items
        merged_mask_ids = set(mask_ids)
        merged_items = list(active_items)
        merged_color = merged_items[0].default_color

         # Flatten the groups and remove them
        for group in unique_groups:
            merged_mask_ids.update(group["mask_ids"])
            merged_items.extend(self.get_items_by_ids(group["mask_ids"]))
            self.connected_groups.remove(group)

        # Create and register a new group
        new_group = {
            "mask_ids": merged_mask_ids,
            "color": merged_color,
            "mask": merged_mask
        }
        self.connected_groups.append(new_group)
        for mid in merged_mask_ids:
            self.mask_id_to_group[mid] = new_group

        # Update callbacks and recolor masks
        for item in active_items:
            self.recolor_mask(item, merged_color)
            key = f"{item.name}_{item.label}"

            if key not in self.pre_merge_callback_state:
                self.pre_merge_callback_state[key] = self.callback_dict.get(key, {}).copy()

            if key in self.callback_dict:
                self.callback_dict[key]["merged"] = True
            else:
                self.callback_dict[key] = {
                    "name": item.name,
                    "label": item.label,
                    "is_active": True,
                    "merged": False
                }

    def get_item_by_id(self, mask_id):
        """Return the mask item corresponding to the given mask ID."""
        for item in self.mask_items:
            if self.get_mask_id(item.name, item.label) == mask_id:
                return item
        return None
    
    def get_items_by_ids(self, id_set):
            """Return a list of mask items for a given set of mask IDs."""
            return [self.get_item_by_id(mid) for mid in id_set]
    
    def get_mask_id(self, name, label):
        """Generate a unique ID string for a mask from its name and label."""
        return f"{name}_{label}"
    
    def disconnect_mask(self, mask_id):
        """Disconnect an individual mask from its group and restore its original state."""
        group = self.mask_id_to_group.get(mask_id)
        if not group:
            return

        disconnected_item = self.get_item_by_id(mask_id)
        if disconnected_item is None:
            return

        merged_key = self.get_merged_key_from_group(group)
        self.remove_merged_mask_entries(merged_key)

        for remaining_id in group["mask_ids"]:
            self.restore_individual_mask(remaining_id)

        if len(group["mask_ids"]) < 2:
            self.remove_group(group)

            for remaining_id in list(group["mask_ids"]):
                if remaining_id == mask_id:
                    continue
                self.restore_individual_mask(remaining_id)

        self.remove_merged_mask_entries(merged_key)
        self.remove_group(group)
        self.refresh_scene()

    def disconnect_group(self, group):
        """Disconnect all masks in a group and restore them individually."""
        merged_key = self.get_merged_key_from_group(group)
        self.remove_merged_mask_entries(merged_key)

        for mid in list(group["mask_ids"]):
            self.restore_individual_mask(mid)

        self.remove_group(group)
        self.refresh_scene()

    def get_merged_key_from_group(self, group):
        """Generate a key to identify the merged mask from a group."""
        label_names = sorted(
            str(self.get_item_by_id(mid).label)
            for mid in group["mask_ids"]
            if self.get_item_by_id(mid)
        )
        name = next(iter(group["mask_ids"])).split("_")[0]
        return f"{name}_({','.join(label_names)})"

    def remove_merged_mask_entries(self, merged_key):
        """Remove merged mask metadata from internal tracking dictionaries."""
        self.callback_dict.pop(merged_key, None)
        self.new_mask_dict.pop(merged_key, None)

    def restore_individual_mask(self, mid):
        """Restore a single mask to its pre-merge state."""
        item = self.get_item_by_id(mid)
        if not item:
            return

        key = f"{item.name}_{item.label}"
        cached = self.pre_merge_callback_state.get(key)
        self.callback_dict[key] = {
            "name": item.name,
            "label": item.label,
            "is_active": cached["is_active"] if cached else True,
            "merged": False
        }

        self.new_mask_dict[key] = {
            "mask": item.binary_mask,
            "source": "disconnect",
            "image_name": item.name,
            "label_group": [str(item.label)],
        }
        # Apply original color and make visible
        self.recolor_mask(item, item.default_color)
        item.setOpacity(self.active_opacity)
        item.is_inactive = False

        self.mask_id_to_group.pop(mid, None)

    def remove_group(self, group):
        """Remove a mask group from tracking."""
        if group in self.connected_groups:
            self.connected_groups.remove(group)

    def refresh_scene(self):
        """Force a refresh of the scene and viewport to reflect changes."""
        self.scene().update()
        self.viewport().update()

    def recolor_mask(self, mask_item_to_update, color):
        """Apply a new color to the mask item and update its pixmap."""
        for mask_data in self.original_masks:
            if mask_data["image_name"] == mask_item_to_update.name and mask_data["label"] == mask_item_to_update.label:
                binary_mask = mask_data["mask"]
                rgba_color = (color[0], color[1], color[2], 200)  # force alpha to 200
                new_pixmap = self.convert_mask_to_pixmap(binary_mask, rgba_color)
                 # Resize pixmap to match base image dimensions
                scaled_pixmap = new_pixmap.scaled(
                    self.pixmap_item.pixmap().width(),
                    self.pixmap_item.pixmap().height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                mask_item_to_update.setPixmap(scaled_pixmap)
                return

    def set_togglable_masks(self, masks, colors, pixmap, font_size, show_labels=False):
        """
        Making each mask a togglable object with assigned properties, centered at its object centroid.
        """
        # Clear existing masks and state
        for item in self.mask_items:
            self.graphics_scene.removeItem(item)
        self.mask_items.clear()
        self.connected_groups.clear()
        self.mask_id_to_group.clear()
        # Add and register new masks
        for i, mask_data in enumerate(masks):
            label = mask_data["label"]
            name = mask_data["image_name"]
            mask = mask_data["mask"]  # 2D numpy binary mask
            color = colors[i]

            key = f"{name}_{label}"
            if key not in self.callback_dict:
                self.callback_dict[key] = {
                    "name": name,
                    "label": label,
                    "is_active": True,
                    "merged": False
                }
            # Convert mask to pixmap and scale it to the image size
            mask_pixmap = self.convert_mask_to_pixmap(mask, color)
            scaled_pixmap = mask_pixmap.scaled(
                pixmap.width(),
                pixmap.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            # Create interactive mask object
            mask_item = ClickableMask(
                scaled_pixmap, name, label, self.mask_click_callback, binary_mask=mask,
                connection_mode_getter=self.is_connection_mode,
                viewer=self
            )

            # Store original color and add to scene
            mask_item.default_color = color
            mask_item.setZValue(1)
            mask_item.setPos(0, 0)
            self.graphics_scene.addItem(mask_item)
            self.mask_items.append(mask_item)

            if key not in self.new_mask_dict:
                self.new_mask_dict[key] = {
                    "mask": mask,
                    "source": "individual",
                    "image_name": name,
                    "label_group": label
                }
            # If applicable, add a label at mask centroid
            if show_labels:
                y_coords, x_coords = np.nonzero(mask)
                if len(x_coords) == 0 or len(y_coords) == 0:
                    continue  # Skip empty masks

                # Compute centroid in original mask coordinates
                centroid_x = np.mean(x_coords)
                centroid_y = np.mean(y_coords)

                # Scale centroid to match the scaled pixmap
                scale_x = scaled_pixmap.width() / mask.shape[1]
                scale_y = scaled_pixmap.height() / mask.shape[0]
                scaled_x = centroid_x * scale_x
                scaled_y = centroid_y * scale_y

                label_item = QGraphicsTextItem(str(label))
                font = QFont("Arial", font_size if font_size is not None else 12)
                font.setWeight(QFont.Weight.Bold)
                label_item.setFont(font)
                label_item.setDefaultTextColor(Qt.GlobalColor.white)
                label_item.setZValue(100)

                rect = label_item.boundingRect()
                label_item.setPos(scaled_x - rect.width() / 2, scaled_y - rect.height() / 2)
                self.graphics_scene.addItem(label_item)

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
        self.callback_dict[f"{name}_{label}"] = {'name': name, 'label': label, 'is_active': is_active}  
