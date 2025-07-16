import numpy as np
import cv2
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem
from PyQt6.QtGui import QPixmap, QImage, QColor, QFont
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen, QPainter
from PyQt6.QtWidgets import QGraphicsPathItem
from PyQt6.QtGui import QPainterPath

from skimage import measure

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
        print(f"Mouse press in {self.name}_{self.label}, viewer mode: {self.viewer.mode}, connection_mode_getter: {self.connection_mode_getter()}")

        # Check both connection_mode_getter and viewer.mode to ensure consistency
        if self.connection_mode_getter() or self.viewer.mode != "toggle":
            event.ignore()
            return


        if self.viewer.mode == "toggle":
            # Use group toggle if in group
            mask_id = self.viewer.get_mask_id(self.name, self.label)
            group = self.viewer.mask_id_to_group.get(mask_id)

            if group:
                # Determine new state from current item
                turning_off = self.opacity() == self.viewer.active_opacity
                merged_key = self.viewer.generate_merged_key(self.name, group["mask_ids"])
                
                # Update callback_dict for the merged mask
                self.viewer.callback_dict[merged_key] = {
                    "name": self.name,
                    "label": merged_key.split('_', 1)[1],  # Extract the merged label
                    "is_active": not turning_off,
                    "merged": True
                }

                # Update individual masks in the group
                for mid in group["mask_ids"]:
                    item = self.viewer.get_item_by_id(mid)
                    if item:
                        item.setOpacity(self.viewer.inactive_opacity if turning_off else self.viewer.active_opacity)
                        item.is_inactive = turning_off
                        # Remove or mark individual masks as inactive in callback_dict
                        individual_key = f"{item.name}_{item.label}"
                        if turning_off:
                            if individual_key in self.viewer.callback_dict:
                                self.viewer.callback_dict[individual_key]["is_active"] = False
                        else:
                            if individual_key in self.viewer.callback_dict:
                                self.viewer.callback_dict[individual_key]["is_active"] = True

                if turning_off and merged_key in self.viewer.new_mask_dict:
                    print(f"🗑️ Toggled OFF — removing {merged_key} from new_mask_dict")
                    del self.viewer.new_mask_dict[merged_key]

            else:
                # Fallback: toggle just self
                turning_off = self.opacity() == self.viewer.active_opacity
                self.setOpacity(self.viewer.inactive_opacity if turning_off else self.viewer.active_opacity)
                self.is_inactive = turning_off
                self.viewer.mask_click_callback(self.name, self.label, not turning_off)

            event.accept()

        

class ImageViewer(QGraphicsView):
    def __init__(self, pixmap, masks, font_size, show_labels=False, colors=None, image_name=None, worker=None):
        super().__init__()

        self.callback_dict = {} # Collect callback functions for each mask
        self.connection_mode = False
        self.disconnect_mode = False  # Explicitly initialize
        self.mouse_path = []
        self.connection_line = None
        self.original_masks = masks
        self.connected_groups = []  # each group is a dict: {'mask_ids': set, 'color': tuple, 'star_item': QGraphicsTextItem}
        self.mask_id_to_group = {}  # for quick lookup: mask_id -> group dict
        self.disconnect_mode = False
        self.mode = "toggle"  # default mode
        self.new_mask_dict = {}
        self.correction_action = "connect"  # or "disconnect", depending on your UI
        self.drawing = False
        self.last_draw_point = None
        self.image_name = image_name
        self.drawing_canvas = QPixmap(pixmap.size())
        self.drawing_canvas.fill(Qt.GlobalColor.transparent)
        self.drawing_item = QGraphicsPixmapItem(self.drawing_canvas)
        self.drawing_item.setZValue(999)  # On top of masks
        self.active_opacity = 1.0
        self.inactive_opacity = 0.2
        self.worker = worker
        self.pre_merge_callback_state = {}
        self.set_mode("toggle")

        # Disable scroll bars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setStyleSheet("border: none; padding: 0px; margin: 0px;")

        # Set up a container for the image
        self.graphics_scene = QGraphicsScene(self)
        self.setScene(self.graphics_scene)
        self.graphics_scene.addItem(self.drawing_item)
        self.setFixedSize(pixmap.size())
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.graphics_scene.addItem(self.pixmap_item)

        self.mask_items = [] # For storing mask items added to the scene

        # Generating a unique color for each mask
        num_masks = len(masks)
        colors = self.generate_colors(num_masks)
        self.set_togglable_masks(masks, colors, pixmap, font_size=font_size, show_labels=show_labels)
        
    def generate_merged_key(self, name, mask_ids):
        labels = sorted(str(self.get_item_by_id(mid).label) for mid in mask_ids)
        return  f"{name}({','.join(labels)})"
   
  
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_C:
            self.set_mode("connect")
            self.correction_action = "connect"
            print("Switched to connect mode → CONNECT")

        elif event.key() == Qt.Key.Key_D:
            self.set_mode("connect")
            self.correction_action = "disconnect"
            print("Switched to connect mode → DISCONNECT")

        elif event.key() == Qt.Key.Key_T:
            self.set_mode("toggle")
            print("Switched to TOGGLE mode")



    def is_connection_mode(self):
        return self.connection_mode
    
    def mousePressEvent(self, event):
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
        if self.mode == "connect" and event.buttons() & Qt.MouseButton.LeftButton:
            scene_point = self.mapToScene(event.position().toPoint())
            self.mouse_path.append(scene_point)

            # Draw stroke line as before
            if self.connection_line:
                self.graphics_scene.removeItem(self.connection_line)

            path = QPainterPath()
            path.moveTo(self.mouse_path[0])
            for pt in self.mouse_path[1:]:
                path.lineTo(pt)

            pen = QPen(QColor("white"))
            pen.setWidth(2)
            self.connection_line = QGraphicsPathItem(path)
            self.connection_line.setPen(pen)
            self.connection_line.setZValue(200)
            self.graphics_scene.addItem(self.connection_line)
            event.accept()

        elif self.mode == "draw" and self.drawing:
            painter = QPainter(self.drawing_canvas)
            pen = QPen(QColor("red"), 3, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            current_point = self.mapToScene(event.position().toPoint())
            painter.drawLine(self.last_draw_point, current_point)
            painter.end()

            self.drawing_item.setPixmap(self.drawing_canvas)
            self.last_draw_point = current_point
            event.accept()

        elif self.mode == "erase" and self.drawing:
            painter = QPainter(self.drawing_canvas)
            # Set composition mode to clear to erase
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            eraser_size = 30  # or any size you want
            pen = QPen(QColor(0, 0, 0, 0), eraser_size)
            painter.setPen(pen)
            current_point = self.mapToScene(event.position().toPoint())
            painter.drawLine(self.last_draw_point, current_point)
            painter.end()

            self.drawing_item.setPixmap(self.drawing_canvas)
            self.last_draw_point = current_point
            event.accept()

        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):

        if self.mode == "connect" and event.button() == Qt.MouseButton.LeftButton:
            print(f"Mouse released in connect mode, action: {self.correction_action}")
            self.mouse_path.append(self.mapToScene(event.position().toPoint()))
            self.handle_mouse_path()
            self.mouse_path = []
            if self.connection_line:
                self.graphics_scene.removeItem(self.connection_line)
                self.connection_line = None
            event.accept()
        elif self.mode in ("draw", "erase") and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.last_draw_point = None
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
        if mode not in ("toggle", "connect", "draw", "erase"):
            print(f"Invalid mode: {mode}")
            return
        self.mode = mode
        self.connection_mode = (mode == "connect")
        self.disconnect_mode = (mode == "connect")
        print(f"Mode switched to {mode}")


    def is_connection_mode(self):
        # Only connection mode active if in connect mode
        return self.mode == "connect"
   
    def handle_mouse_path(self):

        hit_masks = set()
        for item in self.mask_items: # mask_items are all currently displayed masks objects
            if item.is_inactive:
                continue  # Skip masks that are toggled off
            for pt in self.mouse_path:
                scene_x, scene_y = pt.x(), pt.y()

                # mapping mouse position on the screen to actual pixel positions inside the mask array
                img_x = int(scene_x * item.binary_mask.shape[1] / self.pixmap_item.pixmap().width())
                img_y = int(scene_y * item.binary_mask.shape[0] / self.pixmap_item.pixmap().height())

                if 0 <= img_x < item.binary_mask.shape[1] and 0 <= img_y < item.binary_mask.shape[0]:
                    if item.binary_mask[img_y, img_x]:
                        hit_masks.add(item)
                        break
        
        active_items = [m for m in hit_masks if not m.is_inactive]
      

        if len(active_items) > 1:
      
            merged_mask = np.zeros_like(active_items[0].binary_mask, dtype=np.int32)
            label_names = []

            for item in active_items:
                merged_mask[item.binary_mask > 0] = 1
                relabeled_mask = measure.label(merged_mask)
                label_names.append(str(item.label))

            label_names = sorted(label_names)
            combined_label = "_".join(label_names)
            name = active_items[0].name
            mask_key = f"{name}_({combined_label})"

            self.new_mask_dict[mask_key] = {
                "mask": merged_mask,
                "source": "connect",
                "image_name": name,
                "label_group": label_names
            }
            self.callback_dict[mask_key] = {
                "name": name,
                "label": mask_key,  # the combined label string
                "is_active": True,
                "merged": True
            }

        else:
            return

        

        mask_ids = {self.get_mask_id(m.name, m.label) for m in active_items}
        merged_mask_ids = set(mask_ids)  # define it here early
        connected_mask_ids = set()
        for m in hit_masks:
            mask_id = self.get_mask_id(m.name, m.label)
            if mask_id in self.mask_id_to_group:
                connected_mask_ids.add(mask_id)

        if len(connected_mask_ids) == len(hit_masks):
            group_ids = [self.mask_id_to_group[mid] for mid in connected_mask_ids]
            if len(set(map(id, group_ids))) > 1:
                return

            print("Auto-disconnecting group")
          

            group = self.mask_id_to_group.get(next(iter(connected_mask_ids)))
            if group:
                self.disconnect_group(group)


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
                self.connected_groups.remove(group)
        new_group = {
            "mask_ids": merged_mask_ids,
            "color": merged_color,
            'mask': merged_mask
        }
        self.connected_groups.append(new_group)

        for mid in merged_mask_ids:
            self.mask_id_to_group[mid] = new_group

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
        for item in self.mask_items:
            if self.get_mask_id(item.name, item.label) == mask_id:
                return item
        return None
    
    def get_items_by_ids(self, id_set):
            return [self.get_item_by_id(mid) for mid in id_set]
    
    def get_mask_id(self, name, label):
        return f"{name}_{label}"

    def disconnect_mask(self, mask_id):
        group = self.mask_id_to_group.get(mask_id)
        if not group:
            print("Not in a group.")
            return

        # ✅ Now group is always defined beyond this point
        disconnected_item = self.get_item_by_id(mask_id)
        if disconnected_item is None:
            print(f"⚠️ No item found for {mask_id}")
            return

        label_names = sorted(
            str(self.get_item_by_id(mid).label)
            for mid in group["mask_ids"]
            if self.get_item_by_id(mid)
        )
        name = disconnected_item.name
        merged_key = f"{name}_({','.join(label_names)})"

    
        if merged_key in self.callback_dict:
            del self.callback_dict[merged_key]
        if merged_key in self.new_mask_dict:
            del self.new_mask_dict[merged_key]

        for remaining_id in group["mask_ids"]:
            item = self.get_item_by_id(remaining_id)
            if not item:
                continue

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


        self.connected_groups = [g for g in self.connected_groups if g != group]
 
    
        # Step 2: If group has < 2 remaining → dissolve it fully
        if len(group["mask_ids"]) < 2:
            self.connected_groups.remove(group)

            for remaining_id in list(group["mask_ids"]):  # even if it's just 1
                
                item = self.get_item_by_id(remaining_id)
                if item:
                    self.recolor_mask(item, item.default_color)
                    item.setOpacity(self.active_opacity)
                    item.is_inactive = False

                    # 🔴 This may be missing:
                    if remaining_id in self.mask_id_to_group:
                        del self.mask_id_to_group[remaining_id]

                    if remaining_id == mask_id:
                        continue 

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

        # Step 3: Refresh scene
        if merged_key in self.new_mask_dict:
            del self.new_mask_dict[merged_key]
        if group in self.connected_groups:
            self.connected_groups.remove(group)
        self.scene().update()
        self.viewport().update()

    def disconnect_group(self, group):

        # Generate merged key (before group is dissolved)
        label_names = sorted(
            str(self.get_item_by_id(mid).label)
            for mid in group["mask_ids"]
            if self.get_item_by_id(mid)
        )
        name = next(iter(group["mask_ids"])).split("_")[0]  # crude way to extract name
        merged_key = f"{name}_({','.join(label_names)})"

   
        if merged_key in self.callback_dict:
            del self.callback_dict[merged_key]
        if merged_key in self.new_mask_dict:
            del self.new_mask_dict[merged_key]
        

        for mid in list(group["mask_ids"]):
            item = self.get_item_by_id(mid)
            if not item:
                continue
            self.recolor_mask(item, item.default_color)
            item.setOpacity(self.active_opacity)
            item.is_inactive = False

            if mid in self.mask_id_to_group:
                del self.mask_id_to_group[mid]

            key = f"{item.name}_{item.label}"
            self.callback_dict[key] = {
                "name": item.name,
                "label": item.label,
                "is_active": True,
                "merged": False
            }

            self.new_mask_dict[key] = {
                "mask": item.binary_mask,
                "source": "disconnect",
                "image_name": item.name,
                "label_group": [str(item.label)],
            }

        
        if group in self.connected_groups:
            self.connected_groups.remove(group)
        
        self.scene().update()
        self.viewport().update()




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

        for item in self.mask_items:
            self.graphics_scene.removeItem(item)
        self.mask_items.clear()
        self.connected_groups.clear()
        self.mask_id_to_group.clear()
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

            # Add the mask on top of the image at (0, 0)
            # Before adding the mask to the scene
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

            # ✅ Save individual mask to new_mask_dict
            if key not in self.new_mask_dict:
                self.new_mask_dict[key] = {
                    "mask": mask,
                    "source": "individual",
                    "image_name": name,
                    "label_group": label
                }

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
        self.callback_dict[f"{name}_{label}"] = {'name': name, 'label': label, 'is_active': is_active}  

