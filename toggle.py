import numpy as np
import cv2
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem
from PyQt6.QtGui import QPixmap, QImage, QColor, QFont
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPen
from PyQt6.QtWidgets import QGraphicsPathItem
from PyQt6.QtGui import QPainterPath
from PyQt6.QtWidgets import QMessageBox

class ClickableMask(QGraphicsPixmapItem):
    """
    Allows toggling of the mask opacity when it's clicked
    """
    def __init__(self, pixmap, name, label, click_callback, binary_mask, connection_mode_getter, viewer):
        super().__init__(pixmap)
        self.setAcceptHoverEvents(True)
       
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.active_opacity = 1.0
        self.inactive_opacity = 0.2
        self.setOpacity(self.active_opacity)  # Start fully visible
        self.label = label  # Store the label for this mask
        self.name = name # Store the name for this mask
        self.is_inactive = False  # Track if mask is dimmed
        self.binary_mask = binary_mask
        self.connection_mode_getter = connection_mode_getter
        self.click_callback = click_callback  # Store the callback function
        self.viewer = viewer

    def mousePressEvent(self, event):
        if self.viewer.mode == "correction":
            # Disable toggling during correction mode (stroke connect/disconnect only)
            print("Toggle disabled during correction mode")
            event.ignore()
            return

        # Existing toggle logic for toggle mode
        if self.opacity() == self.active_opacity:
            self.setOpacity(self.inactive_opacity)
            self.is_inactive = True
        else:
            self.setOpacity(self.active_opacity)
            self.is_inactive = False

        self.click_callback(self.name, self.label, not self.is_inactive)
        event.accept()

class ImageViewer(QGraphicsView):
    def __init__(self, pixmap, masks, font_size, show_labels=False, colors=None):
        super().__init__()

        self.callback_dict = {} # Collect callback functions for each mask
        self.connection_mode = False
        self.mouse_path = []
        self.connection_line = None
        self.connected_pairs = set()
        self.original_masks = masks
        self.connected_groups = []  # each group is a dict: {'mask_ids': set, 'color': tuple, 'star_item': QGraphicsTextItem}
        self.mask_id_to_group = {}  # for quick lookup: mask_id -> group dict
        self.disconnect_mode = False
        self.mode = "toggle"  # default mode
        self.correction_action = "connect"  # or "disconnect", depending on your UI





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
        self.set_togglable_masks(masks, colors, pixmap, font_size=font_size, show_labels=show_labels)
    
    def get_mask_id(self, name, label):
        return f"{name}_{label}"
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_C:
            self.set_mode("correction")
            self.correction_action = "connect"
            print("Switched to correction mode → CONNECT")

        elif event.key() == Qt.Key.Key_D:
            self.set_mode("correction")
            self.correction_action = "disconnect"
            print("Switched to correction mode → DISCONNECT")

        elif event.key() == Qt.Key.Key_T:
            self.set_mode("toggle")
            print("Switched to TOGGLE mode")



    def is_connection_mode(self):
        return self.connection_mode
    
    def mousePressEvent(self, event):
        if self.mode == "correction":
            # Accept and start stroke
            self.mouse_path = [self.mapToScene(event.position().toPoint())]
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.mode == "correction" and event.buttons() & Qt.MouseButton.LeftButton:
            scene_point = self.mapToScene(event.position().toPoint())
            self.mouse_path.append(scene_point)

            # Draw stroke line as before
            if self.connection_line:
                self.scene.removeItem(self.connection_line)

            path = QPainterPath()
            path.moveTo(self.mouse_path[0])
            for pt in self.mouse_path[1:]:
                path.lineTo(pt)

            pen = QPen(QColor("white"))
            pen.setWidth(2)
            self.connection_line = QGraphicsPathItem(path)
            self.connection_line.setPen(pen)
            self.connection_line.setZValue(200)
            self.scene.addItem(self.connection_line)

            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.mode == "correction" and event.button() == Qt.MouseButton.LeftButton:
            print(f"Mouse released in correction mode, action: {self.correction_action}")
            self.mouse_path.append(self.mapToScene(event.position().toPoint()))
            self.handle_mouse_path()
            self.mouse_path = []
            if self.connection_line:
                self.scene.removeItem(self.connection_line)
                self.connection_line = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)




    def extract_color_from_pixmap(self, pixmap):
        image = pixmap.toImage()
        width, height = image.width(), image.height()
        
        # Scan for first non-transparent pixel
        for y in range(height):
            for x in range(width):
                pixel = image.pixelColor(x, y)
                if pixel.alpha() > 0:
                    return (pixel.red(), pixel.green(), pixel.blue(), pixel.alpha())
        
        return (255, 255, 255, 255)  # fallback white
    def set_mode(self, mode):
        """
        mode: 'toggle' or 'correction'
        """
        if mode not in ("toggle", "correction"):
            print(f"Invalid mode: {mode}")
            return

        self.mode = mode
        self.connection_mode = (mode == "correction")  # connection mode only ON in correction
        self.disconnect_mode = (mode == "correction")  # allow disconnect in correction mode

        print(f"Mode switched to {mode}")

    def is_connection_mode(self):
        # Only connection mode active if in correction mode
        return self.mode == "correction"
   
    def handle_mouse_path(self):
        hit_masks = set()
        for item in self.mask_items:
            if item.is_inactive:
                continue  # Skip masks that are toggled off
            for pt in self.mouse_path:
                scene_x, scene_y = pt.x(), pt.y()

                # Convert scene coordinates to image mask coordinates
                img_x = int(scene_x * item.binary_mask.shape[1] / self.pixmap_item.pixmap().width())
                img_y = int(scene_y * item.binary_mask.shape[0] / self.pixmap_item.pixmap().height())

                if 0 <= img_x < item.binary_mask.shape[1] and 0 <= img_y < item.binary_mask.shape[0]:
                    if item.binary_mask[img_y, img_x]:
                        hit_masks.add(item)
                        break

        if len(hit_masks) < 2:
            print(f"Need at least 2 active masks (got {len(hit_masks)}).")
            return

        mask_ids = {self.get_mask_id(m.name, m.label) for m in hit_masks}

        connected_mask_ids = {self.get_mask_id(m.name, m.label) for m in hit_masks if self.get_mask_id(m.name, m.label) in self.mask_id_to_group}

        if len(connected_mask_ids) == len(hit_masks):
            # All masks are connected —> disconnect them
            group_ids = [self.mask_id_to_group[mid] for mid in connected_mask_ids]
            if len(set(map(id, group_ids))) > 1:
                print("Not all masks are in the same group; can't disconnect.")
                return

            print("Auto-disconnecting group")
            for mid in connected_mask_ids:
                self.disconnect_mask(mid)
            return
        else:
            # Not all masks are connected —> proceed to connect
            print("Auto-connecting masks")


        # === Connection logic ===
        groups = [self.mask_id_to_group[mid] for mid in mask_ids if mid in self.mask_id_to_group]
        seen = set()
        unique_groups = []
        for group in groups:
            group_id = id(group)
            if group_id not in seen:
                seen.add(group_id)
                unique_groups.append(group)

        merged_mask_ids = set(mask_ids)
        merged_items = list(hit_masks)
        merged_color = merged_items[0].default_color

        if unique_groups:
            for group in unique_groups:
                merged_mask_ids.update(group["mask_ids"])
                merged_items.extend(self.get_items_by_ids(group["mask_ids"]))
                self.scene.removeItem(group["star_item"])
                self.connected_groups.remove(group)

        new_group = {
            "mask_ids": merged_mask_ids,
            "color": merged_color,
            "star_item": None,
        }
        self.connected_groups.append(new_group)

        for mid in merged_mask_ids:
            self.mask_id_to_group[mid] = new_group

        for item in merged_items:
            self.recolor_mask(item, merged_color)
            key = f"{item.name}{item.label}"
            if key in self.callback_dict:
                self.callback_dict[key]["merged"] = True
            else:
                self.callback_dict[key] = {
                    "name": item.name,
                    "label": item.label,
                    "is_active": not item.is_inactive,
                    "merged": True
                }


    def get_item_by_id(self, mask_id):
        for item in self.mask_items:
            if self.get_mask_id(item.name, item.label) == mask_id:
                return item
        return None
    
    def show_warning(self, title, message):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()

    def get_items_by_ids(self, id_set):
            return [self.get_item_by_id(mid) for mid in id_set]
    

    def disconnect_mask(self, mask_id):
        print(f"Attempting to disconnect {mask_id}")
        group = self.mask_id_to_group.get(mask_id)
        if not group:
            print("Not in a group.")
            return
        # Get ALL items in this group BEFORE modifying
        all_group_ids = set(group["mask_ids"])
        all_group_items = [self.get_item_by_id(mid) for mid in all_group_ids]

        # Remove mask_id from group
        group["mask_ids"].remove(mask_id)
        del self.mask_id_to_group[mask_id]

        # Recolor the disconnected mask
        disconnected_item = self.get_item_by_id(mask_id)
        if disconnected_item:
            self.recolor_mask(disconnected_item, disconnected_item.default_color)
            key = f"{disconnected_item.name}{disconnected_item.label}"
            if key in self.callback_dict:
                self.callback_dict[key]["merged"] = False
            else:
                self.callback_dict[key] = {
                    "name": disconnected_item.name,
                    "label": disconnected_item.label,
                    "is_active": not disconnected_item.is_inactive,
                    "merged": False
                }

        # If only 1 mask remains, dissolve the group and recolor it
        if len(group["mask_ids"]) < 2:
            if group["star_item"]:
                self.scene.removeItem(group["star_item"])
            self.connected_groups.remove(group)

            for remaining_id in group["mask_ids"]:
                item = self.get_item_by_id(remaining_id)
                if item:
                    self.recolor_mask(item, item.default_color)
                del self.mask_id_to_group[remaining_id]





    def recolor_mask(self, mask_item_to_update, color):
        for mask_data in self.original_masks:
            if mask_data["image_name"] == mask_item_to_update.name and mask_data["label"] == mask_item_to_update.label:
                binary_mask = mask_data["mask"]
                rgba_color = (color[0], color[1], color[2], 200)  # force alpha to 200
                new_pixmap = self.convert_mask_to_pixmap(binary_mask, rgba_color)

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
        image_width = pixmap.width()
        image_height = pixmap.height()

        for i, mask_data in enumerate(masks):
            label = mask_data["label"]
            name = mask_data["image_name"]
            mask = mask_data["mask"]  # 2D numpy binary mask
            color = colors[i]
           

            # Convert mask to pixmap and scale it to the image size
            mask_pixmap = self.convert_mask_to_pixmap(mask, color)
            scaled_pixmap = mask_pixmap.scaled(
                pixmap.width(),
                pixmap.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            # Add the mask on top of the image at (0, 0)
            # Before adding the mask to the scene
            mask_item = ClickableMask(
                scaled_pixmap, name, label, self.mask_click_callback, binary_mask=mask,
                connection_mode_getter=self.is_connection_mode,
                viewer=self  # pass the viewer
            )

            mask_item.default_color = color # <-- store original color


            mask_item.setZValue(1)
            mask_item.setPos(0, 0)
            self.scene.addItem(mask_item)
            self.mask_items.append(mask_item)

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
                self.scene.addItem(label_item)

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


