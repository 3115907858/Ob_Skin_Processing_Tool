import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk, messagebox

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image, ImageDraw, ImageTk, ImageOps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PerformanceTracker:
    def __init__(self):
        self.last_update = 0
        self.update_interval = 1 / 60
        self.points_buffer = []
        self.max_buffer_size = 20
        self.temp_stroke_mask = None
        self.mask_shape = None

    def should_update(self):
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.last_update = current_time
            return True
        return False

    def reset_temp_mask(self, shape):
        if self.temp_stroke_mask is None or self.mask_shape != shape:
            self.mask_shape = shape
            self.temp_stroke_mask = np.zeros(shape, dtype=np.float32)
        else:
            self.temp_stroke_mask.fill(0)
        return self.temp_stroke_mask

class MaskEditor:
    def __init__(self, orig_mask=None, downscale_factor=2.0):
        self.orig_mask_for_editing_layer1 = None
        self.edit_mask_layer1 = None
        self.downscale_factor = downscale_factor
        self.has_changes = False
        self.last_processed_time = 0
        self.min_update_interval = 1 / 30.0
        self.stroke_buffer = []
        self.max_buffer_size = 10
        self.temp_mask = None
        self.display_cache = None
        self.display_cache_valid = False
        self.canvas_size = (0, 0)
        self.neutral_mask_value = 0.5
        self.on_mask_changed = None

    def init_from_mask(self, layer1_mask, neutral_value=0.5):
        if layer1_mask is None:
            return

        self.orig_mask_for_editing_layer1 = layer1_mask.copy()
        self.neutral_mask_value = neutral_value

        h, w = layer1_mask.shape[:2]
        self.downscaled_h = max(50, int(h / self.downscale_factor))
        self.downscaled_w = max(50, int(w / self.downscale_factor))

        self.edit_mask_layer1 = cv2.resize(
            layer1_mask, (self.downscaled_w, self.downscaled_h), interpolation=cv2.INTER_LINEAR
        )

        if self.temp_mask is None or self.temp_mask.shape != self.edit_mask_layer1.shape:
            self.temp_mask = np.zeros_like(self.edit_mask_layer1)

        self.has_changes = False
        self.display_cache_valid = False

    def get_final_mask(self):
        if self.orig_mask_for_editing_layer1 is None or self.edit_mask_layer1 is None:
            return None

        if self.has_changes:
            h, w = self.orig_mask_for_editing_layer1.shape[:2]
            self.orig_mask_for_editing_layer1 = cv2.resize(
                self.edit_mask_layer1, (w, h), interpolation=cv2.INTER_LINEAR
            )
        return self.orig_mask_for_editing_layer1

    def add_stroke(self, x1, y1, x2, y2, brush_size, mode, hardness):
        self.stroke_buffer.append((x1, y1, x2, y2, brush_size, mode, hardness))
        self.has_changes = True
        self.display_cache_valid = False

        if len(self.stroke_buffer) >= self.max_buffer_size:
            self.process_strokes()
            return True

        current_time = time.time()
        if current_time - self.last_processed_time >= self.min_update_interval:
            self.process_strokes()
            return True
        return False

    def process_strokes(self):
        if not self.stroke_buffer or self.edit_mask_layer1 is None:
            return

        if self.temp_mask is None or self.temp_mask.shape != self.edit_mask_layer1.shape:
            self.temp_mask = np.zeros_like(self.edit_mask_layer1, dtype=np.float32)
        else:
            self.temp_mask.fill(0.0)

        current_mode = self.stroke_buffer[0][5]

        if current_mode != "blur":
            for x1, y1, x2, y2, brush_size, mode, hardness in self.stroke_buffer:
                img_x1, img_y1 = self._canvas_to_mask_coords(x1, y1)
                img_x2, img_y2 = self._canvas_to_mask_coords(x2, y2)
                scaled_brush_size = max(1, int(brush_size / self.downscale_factor))

                stroke_shape_mask = np.zeros_like(self.edit_mask_layer1, dtype=np.float32)
                cv2.line(stroke_shape_mask, (img_x1, img_y1), (img_x2, img_y2),
                         1.0, thickness=scaled_brush_size)
                cv2.circle(stroke_shape_mask, (img_x1, img_y1), scaled_brush_size // 2, 1.0, -1)
                cv2.circle(stroke_shape_mask, (img_x2, img_y2), scaled_brush_size // 2, 1.0, -1)

                if hardness < 0.95 and scaled_brush_size > 2:
                    blur_factor = (1.0 - hardness)
                    blur_k_size = max(3, int(blur_factor * scaled_brush_size * 0.7 + scaled_brush_size * 0.15))
                    blur_k_size = blur_k_size + 1 if blur_k_size % 2 == 0 else blur_k_size
                    stroke_shape_mask = cv2.GaussianBlur(stroke_shape_mask, (blur_k_size, blur_k_size), 0)

                self.temp_mask = np.maximum(self.temp_mask, stroke_shape_mask)

            if current_mode == "add":
                self.edit_mask_layer1 = np.maximum(self.edit_mask_layer1, self.temp_mask)
            elif current_mode == "remove":
                self.edit_mask_layer1 = np.minimum(self.edit_mask_layer1, 1.0 - self.temp_mask)

        elif current_mode == "blur":
            blur_path_mask = np.zeros_like(self.edit_mask_layer1, dtype=np.float32)
            stroke_hardness_for_blur = 0.5
            stroke_scaled_brush_size_for_blur = 10

            for x1_b, y1_b, x2_b, y2_b, brush_size_b, _, hardness_b in self.stroke_buffer:
                img_x1_b, img_y1_b = self._canvas_to_mask_coords(x1_b, y1_b)
                img_x2_b, img_y2_b = self._canvas_to_mask_coords(x2_b, y2_b)
                scaled_brush_size_b = max(1, int(brush_size_b / self.downscale_factor))

                stroke_hardness_for_blur = hardness_b
                stroke_scaled_brush_size_for_blur = scaled_brush_size_b

                cv2.line(blur_path_mask, (img_x1_b, img_y1_b), (img_x2_b, img_y2_b), 1.0, thickness=scaled_brush_size_b)
                cv2.circle(blur_path_mask, (img_x1_b, img_y1_b), scaled_brush_size_b // 2, 1.0, -1)
                cv2.circle(blur_path_mask, (img_x2_b, img_y2_b), scaled_brush_size_b // 2, 1.0, -1)

            y_indices, x_indices = np.nonzero(blur_path_mask)
            if len(y_indices) > 0:
                blur_padding = stroke_scaled_brush_size_for_blur
                min_y = max(0, np.min(y_indices) - blur_padding)
                max_y = min(self.edit_mask_layer1.shape[0], np.max(y_indices) + blur_padding + 1)
                min_x = max(0, np.min(x_indices) - blur_padding)
                max_x = min(self.edit_mask_layer1.shape[1], np.max(x_indices) + blur_padding + 1)

                region_to_process = self.edit_mask_layer1[min_y:max_y, min_x:max_x]
                path_mask_roi = blur_path_mask[min_y:max_y, min_x:max_x]

                if region_to_process.size > 0 and region_to_process.shape[0] > 2 and region_to_process.shape[1] > 2:
                    blur_amount_factor = (1.0 - stroke_hardness_for_blur)
                    blur_kernel_s = max(3,
                                        int(blur_amount_factor * stroke_scaled_brush_size_for_blur * 0.6 + stroke_scaled_brush_size_for_blur * 0.1))
                    blur_kernel_s = blur_kernel_s + 1 if blur_kernel_s % 2 == 0 else blur_kernel_s

                    blurred_region_values = cv2.GaussianBlur(region_to_process, (blur_kernel_s, blur_kernel_s), 0)

                    self.edit_mask_layer1[min_y:max_y, min_x:max_x] = \
                        region_to_process * (1 - path_mask_roi) + blurred_region_values * path_mask_roi

        np.clip(self.edit_mask_layer1, 0.0, 1.0, out=self.edit_mask_layer1)

        self.stroke_buffer = []
        self.last_processed_time = time.time()
        self.display_cache_valid = False

        if self.on_mask_changed:
            self.on_mask_changed()

    def get_display_image(self, original_img, canvas_width, canvas_height, auto_mask_layer0_for_context=None):
        if self.edit_mask_layer1 is None or original_img is None:
            return None

        if (self.display_cache_valid and self.canvas_size == (canvas_width, canvas_height)):
            return self.display_cache

        img_bgr = cv2.cvtColor(original_img, cv2.COLOR_BGRA2BGR) if original_img.shape[2] == 4 else original_img.copy()
        img_small = cv2.resize(img_bgr, (self.downscaled_w, self.downscaled_h), interpolation=cv2.INTER_LINEAR)

        base_display_img = img_small.astype(np.float32) / 255.0

        if auto_mask_layer0_for_context is not None:
            layer0_small = cv2.resize(auto_mask_layer0_for_context, (self.downscaled_w, self.downscaled_h),
                                      interpolation=cv2.INTER_LINEAR)
            layer0_context_color = np.array([0.0, 0.0, 1.0])
            layer0_alpha = 0.3

            layer0_overlay = np.zeros_like(base_display_img)
            active_layer0_pixels = layer0_small > 0.1
            for c in range(3):
                layer0_overlay[:, :, c][active_layer0_pixels] = layer0_context_color[c]

            base_display_img = base_display_img * (1 - layer0_alpha * active_layer0_pixels[..., np.newaxis]) + \
                               layer0_overlay * (layer0_alpha * active_layer0_pixels[..., np.newaxis])
            base_display_img = np.clip(base_display_img, 0, 1)

        layer1_viz_input = self.edit_mask_layer1.copy()
        neutral_pixels_in_layer1 = np.isclose(layer1_viz_input, self.neutral_mask_value, atol=0.01)
        layer1_viz_input[neutral_pixels_in_layer1] = 0.0

        mask_u8_layer1 = (np.clip(layer1_viz_input, 0, 1) * 255).astype(np.uint8)
        colored_mask_layer1_jet = cv2.applyColorMap(mask_u8_layer1, cv2.COLORMAP_JET).astype(np.float32) / 255.0

        alpha_for_layer1_jet_viz = 0.6

        blend_alpha_map_for_layer1 = np.zeros_like(self.edit_mask_layer1, dtype=np.float32)
        blend_alpha_map_for_layer1[~neutral_pixels_in_layer1] = alpha_for_layer1_jet_viz

        blended_final_float = base_display_img.copy()
        for i in range(3):
            blended_final_float[:, :, i] = base_display_img[:, :, i] * (1 - blend_alpha_map_for_layer1) + \
                                           colored_mask_layer1_jet[:, :, i] * blend_alpha_map_for_layer1

        blended_final_u8 = (np.clip(blended_final_float, 0, 1) * 255).astype(np.uint8)

        scale = min(canvas_width / self.downscaled_w, canvas_height / self.downscaled_h)
        display_width = int(self.downscaled_w * scale)
        display_height = int(self.downscaled_h * scale)

        if display_width <= 0 or display_height <= 0:
            display_img_resized = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        else:
            display_img_resized = cv2.resize(blended_final_u8, (display_width, display_height),
                                             interpolation=cv2.INTER_NEAREST)

        self.display_cache = display_img_resized
        self.display_cache_valid = True
        self.canvas_size = (canvas_width, canvas_height)
        return display_img_resized

    def _canvas_to_mask_coords(self, canvas_x, canvas_y):
        if self.canvas_size[0] <= 0 or self.canvas_size[1] <= 0 or self.downscaled_w <= 0 or self.downscaled_h <= 0:
            return 0, 0

        scale_w = self.canvas_size[0] / self.downscaled_w
        scale_h = self.canvas_size[1] / self.downscaled_h
        scale = min(scale_w, scale_h) if scale_w > 0 and scale_h > 0 else 1.0

        offset_x = (self.canvas_size[0] - self.downscaled_w * scale) / 2
        offset_y = (self.canvas_size[1] - self.downscaled_h * scale) / 2

        if scale > 1e-6:
            img_x = int((canvas_x - offset_x) / scale)
            img_y = int((canvas_y - offset_y) / scale)
        else:
            img_x, img_y = 0, 0

        img_x = max(0, min(self.downscaled_w - 1, img_x))
        img_y = max(0, min(self.downscaled_h - 1, img_y))
        return img_x, img_y

class NNRSTextureProcessor:
    def __init__(self, wrinkle_threshold=0.15, smoothing_strength=0.7, debug=False,
                 roughness_brightness=0.0, roughness_contrast=1.0,
                 specular_brightness=0.0, specular_contrast=1.0,
                 use_micronormal=False, micronormal_strength=0.5,
                 micronormal_tile_size=1.0, micronormal_mask_mode="inverse",
                 detect_wrinkles_enabled=True,
                 normal_intensity=1.0, normal_mask_mode="none",
                 use_normal_custom_mask=False,
                 enable_auto_mask_blur = False,
                 auto_mask_blur_strength = 0.5,
                 use_roughness_custom_mask = False,
                 roughness_custom_mask_path = None,
                 use_specular_custom_mask = False,
                 specular_custom_mask_path = None):
        self.wrinkle_threshold = wrinkle_threshold
        self.smoothing_strength = smoothing_strength
        self.debug = debug
        self.original_img = None
        self.normal_map = None
        self.roughness_map = None
        self.specular_map = None
        self.adjusted_roughness_map = None
        self.adjusted_specular_map = None
        self.normal_x = None
        self.normal_y = None
        self.normal_z = None
        self.auto_mask_layer0 = None
        self.hand_drawn_mask_layer1 = None
        self.neutral_mask_value = 0.5
        self.wrinkle_mask = None
        self.gradient_magnitude = None
        self.use_wrinkle_custom_mask = False
        self.wrinkle_custom_mask = None
        self.wrinkle_custom_mask_mode = "blend"
        self.processed_normal = None
        self.processed_normal_x = None
        self.processed_normal_y = None
        self.processed_img = None
        self.roughness_brightness = roughness_brightness
        self.roughness_contrast = roughness_contrast
        self.specular_brightness = specular_brightness
        self.specular_contrast = specular_contrast
        self.normal_intensity = normal_intensity
        self.normal_mask_mode = normal_mask_mode
        self.use_normal_custom_mask = use_normal_custom_mask
        self.normal_custom_mask = None
        self.use_micronormal = use_micronormal
        self.micronormal_strength = micronormal_strength
        self.micronormal_tile_size = micronormal_tile_size
        self.micronormal_mask_mode = micronormal_mask_mode
        self.micronormal_path = None
        self.micronormal_img = None
        self.micronormal_x = None
        self.micronormal_y = None
        self.tiled_micronormal_x = None
        self.tiled_micronormal_y = None
        self.custom_micronormal_mask = None
        self.detect_wrinkles_enabled = detect_wrinkles_enabled
        self.enable_auto_mask_blur = enable_auto_mask_blur
        self.auto_mask_blur_strength = auto_mask_blur_strength

        self.use_roughness_custom_mask = use_roughness_custom_mask
        self.roughness_custom_mask_path = roughness_custom_mask_path
        self.roughness_custom_mask = None
        self.use_specular_custom_mask = use_specular_custom_mask
        self.specular_custom_mask_path = specular_custom_mask_path
        self.specular_custom_mask = None

    def load_texture(self, file_path):
        if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None: raise ValueError(f"Could not load image: {file_path}")
        if img.ndim < 3 or img.shape[2] != 4: raise ValueError(f"Image must be RGBA format (4 channels)")
        self.original_img = img
        return img

    def split_channels(self, img=None):
        if img is None: img = self.original_img
        if img is None: return None, None, None
        b, g, r_channel, a_channel = cv2.split(img)
        self.normal_x = (r_channel.astype(np.float32) / 255.0) * 2.0 - 1.0
        self.normal_y = (g.astype(np.float32) / 255.0) * 2.0 - 1.0
        xy_squared = np.clip(self.normal_x ** 2 + self.normal_y ** 2, 0, 1)
        self.normal_z = np.sqrt(1 - xy_squared)
        normal_x_vis = ((self.normal_x + 1.0) / 2.0 * 255.0).astype(np.uint8)
        normal_y_vis = ((self.normal_y + 1.0) / 2.0 * 255.0).astype(np.uint8)
        normal_z_vis = (self.normal_z * 255.0).astype(np.uint8)
        self.normal_map = cv2.merge([normal_x_vis, normal_y_vis, normal_z_vis])
        self.roughness_map = b
        self.specular_map = a_channel
        if self.roughness_map is not None: self.adjusted_roughness_map = self.roughness_map.copy()
        if self.specular_map is not None: self.adjusted_specular_map = self.specular_map.copy()
        return self.normal_map, self.roughness_map, self.specular_map

    def _apply_brightness_contrast(self, image, brightness_factor, contrast_factor, custom_mask=None):
        if image is None: return None

        img_float = image.astype(np.float32) if custom_mask is not None else image

        beta = brightness_factor * 100
        alpha = contrast_factor

        if img_float.dtype == np.uint8:
            adjusted_image_full = cv2.convertScaleAbs(img_float, alpha=alpha, beta=beta)
        else:
            adjusted_image_full = img_float * alpha + beta

        if custom_mask is not None and custom_mask.shape == img_float.shape[:2]:
            mask_float = custom_mask.astype(np.float32)
            if np.max(mask_float) > 1.0:
                mask_float /= 255.0

            if img_float.ndim == 3 and mask_float.ndim == 2:
                mask_float = cv2.cvtColor(mask_float, cv2.COLOR_GRAY2BGR)

            blended_image = img_float * (1.0 - mask_float) + adjusted_image_full * mask_float

            if image.dtype == np.uint8:
                adjusted_image = np.clip(blended_image, 0, 255).astype(np.uint8)
            else:
                adjusted_image = np.clip(blended_image, 0, 1.0 if np.max(image) <= 1.0 else 255.0)
        else:
            adjusted_image = adjusted_image_full

            if image.dtype == np.uint8 and adjusted_image.dtype != np.uint8:
                adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
            elif image.dtype != np.uint8 and adjusted_image.dtype != image.dtype:
                adjusted_image = np.clip(adjusted_image, 0, np.max(image))

        return adjusted_image

    def adjust_roughness_specular(self):
        if self.use_roughness_custom_mask and self.roughness_custom_mask_path and self.roughness_custom_mask is None:
            try:
                mask_img = cv2.imread(self.roughness_custom_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    self.roughness_custom_mask = mask_img
            except Exception as e:
                print(f"Warning: Could not load roughness custom mask: {e}")
                self.roughness_custom_mask = None

        if self.use_specular_custom_mask and self.specular_custom_mask_path and self.specular_custom_mask is None:
            try:
                mask_img = cv2.imread(self.specular_custom_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    self.specular_custom_mask = mask_img
            except Exception as e:
                print(f"Warning: Could not load specular custom mask: {e}")
                self.specular_custom_mask = None


        if self.roughness_map is not None:
            active_roughness_mask = self.roughness_custom_mask if self.use_roughness_custom_mask else None
            if active_roughness_mask is not None and active_roughness_mask.shape != self.roughness_map.shape[:2]:
                active_roughness_mask = cv2.resize(active_roughness_mask, (self.roughness_map.shape[1], self.roughness_map.shape[0]), interpolation=cv2.INTER_LINEAR)

            self.adjusted_roughness_map = self._apply_brightness_contrast(
                self.roughness_map, self.roughness_brightness, self.roughness_contrast,
                custom_mask=active_roughness_mask
            )
        if self.specular_map is not None:
            active_specular_mask = self.specular_custom_mask if self.use_specular_custom_mask else None
            if active_specular_mask is not None and active_specular_mask.shape != self.specular_map.shape[:2]:
                active_specular_mask = cv2.resize(active_specular_mask, (self.specular_map.shape[1], self.specular_map.shape[0]), interpolation=cv2.INTER_LINEAR)

            self.adjusted_specular_map = self._apply_brightness_contrast(
                self.specular_map, self.specular_brightness, self.specular_contrast,
                custom_mask=active_specular_mask
            )

    def detect_wrinkles(self):
        shape_fallback = (256, 256)
        current_shape = self.normal_x.shape if self.normal_x is not None else \
            (self.original_img.shape[:2] if self.original_img is not None else shape_fallback)

        if not self.detect_wrinkles_enabled:
            self.auto_mask_layer0 = np.zeros(current_shape, dtype=np.float32)
            self.gradient_magnitude = np.zeros(current_shape, dtype=np.float32) if self.auto_mask_layer0 is None else np.zeros_like(self.auto_mask_layer0, dtype=np.float32)
            self._combine_mask_layers()
            return self.wrinkle_mask

        if self.normal_x is None or self.normal_y is None:
            self.auto_mask_layer0 = np.zeros(current_shape, dtype=np.float32)
            self.gradient_magnitude = np.zeros_like(self.auto_mask_layer0, dtype=np.float32)
            self._combine_mask_layers()
            return self.wrinkle_mask

        grad_x_x = cv2.Sobel(self.normal_x, cv2.CV_32F, 1, 0, ksize=3)
        grad_x_y = cv2.Sobel(self.normal_x, cv2.CV_32F, 0, 1, ksize=3)
        grad_y_x = cv2.Sobel(self.normal_y, cv2.CV_32F, 1, 0, ksize=3)
        grad_y_y = cv2.Sobel(self.normal_y, cv2.CV_32F, 0, 1, ksize=3)
        gradient_x = cv2.magnitude(grad_x_x, grad_x_y)
        gradient_y = cv2.magnitude(grad_y_x, grad_y_y)
        gradient_magnitude_auto = np.maximum(gradient_x, gradient_y)

        kernel_size = 5
        mean_x = cv2.blur(self.normal_x, (kernel_size, kernel_size))
        mean_y = cv2.blur(self.normal_y, (kernel_size, kernel_size))
        diff_x = self.normal_x - mean_x
        diff_y = self.normal_y - mean_y
        variance_x = cv2.blur(diff_x * diff_x, (kernel_size, kernel_size))
        variance_y = cv2.blur(diff_y * diff_y, (kernel_size, kernel_size))
        std_dev = np.sqrt(np.maximum(variance_x + variance_y, 0))
        combined_metric = gradient_magnitude_auto * 0.6 + std_dev * 0.4
        if np.any(combined_metric):
            combined_metric = cv2.normalize(combined_metric, None, 0, 1, cv2.NORM_MINMAX)
        else:
            combined_metric = np.zeros_like(combined_metric, dtype=np.float32)

        self.gradient_magnitude = combined_metric

        mean_val = float(np.mean(combined_metric))
        std_val = float(np.std(combined_metric))
        adaptive_threshold = mean_val + self.wrinkle_threshold * (std_val if std_val >= 1e-6 else 0.1)

        detected_mask_raw = (combined_metric > adaptive_threshold).astype(np.float32)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        detected_mask_raw = cv2.morphologyEx(detected_mask_raw, cv2.MORPH_CLOSE, kernel)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        detected_mask_raw = cv2.dilate(detected_mask_raw, kernel_dilate, iterations=1)
        detected_mask_raw = cv2.GaussianBlur(detected_mask_raw, (11, 11), 0)
        if np.any(detected_mask_raw):
            detected_mask_raw = np.power(detected_mask_raw, 0.7)
        else:
            detected_mask_raw = np.zeros_like(detected_mask_raw, dtype=np.float32)

        if self.enable_auto_mask_blur and self.auto_mask_blur_strength > 0:
            blur_k_size = max(3, int(self.auto_mask_blur_strength * 100) + 1)
            blur_k_size = blur_k_size + 1 if blur_k_size % 2 == 0 else blur_k_size
            if blur_k_size > 1:
                detected_mask_raw = cv2.GaussianBlur(detected_mask_raw, (blur_k_size, blur_k_size), 0)

        self.auto_mask_layer0 = detected_mask_raw.copy()

        if self.use_wrinkle_custom_mask and self.wrinkle_custom_mask is not None:
            custom_m = self.wrinkle_custom_mask
            if custom_m.shape != self.auto_mask_layer0.shape:
                custom_m = cv2.resize(custom_m, (self.auto_mask_layer0.shape[1], self.auto_mask_layer0.shape[0]),
                                      cv2.INTER_LINEAR)

            if self.wrinkle_custom_mask_mode == "replace":
                self.auto_mask_layer0 = custom_m.copy()
            elif self.wrinkle_custom_mask_mode == "multiply":
                self.auto_mask_layer0 *= custom_m
            elif self.wrinkle_custom_mask_mode == "subtract":
                self.auto_mask_layer0 = np.maximum(0, self.auto_mask_layer0 - custom_m)
            else:
                self.auto_mask_layer0 = np.clip((self.auto_mask_layer0 + custom_m) / 2.0, 0.0, 1.0)

        self._combine_mask_layers()
        return self.wrinkle_mask

    def set_hand_drawn_mask_layer1(self, hand_drawn_mask, neutral_value=0.5):
        self.neutral_mask_value = neutral_value
        if hand_drawn_mask is None:
            self.hand_drawn_mask_layer1 = None
        else:
            target_shape = self.auto_mask_layer0.shape if self.auto_mask_layer0 is not None else \
                (self.original_img.shape[:2] if self.original_img is not None else None)
            if target_shape and hand_drawn_mask.shape != target_shape:
                self.hand_drawn_mask_layer1 = cv2.resize(
                    hand_drawn_mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                self.hand_drawn_mask_layer1 = hand_drawn_mask.copy()
        self._combine_mask_layers()

    def _combine_mask_layers(self):
        if self.auto_mask_layer0 is None:
            if self.hand_drawn_mask_layer1 is not None:
                max_deviation_from_neutral = self.neutral_mask_value
                if max_deviation_from_neutral < 1e-6:
                    max_deviation_from_neutral = 0.5

                self.wrinkle_mask = np.abs(
                    self.hand_drawn_mask_layer1 - self.neutral_mask_value) / max_deviation_from_neutral
                self.wrinkle_mask = np.clip(self.wrinkle_mask, 0.0, 1.0)
            else:
                if self.original_img is not None:
                    self.wrinkle_mask = np.zeros(self.original_img.shape[:2], dtype=np.float32)
                else:
                    self.wrinkle_mask = None
            return

        combined_mask = self.auto_mask_layer0.copy()

        if self.hand_drawn_mask_layer1 is not None:
            m1 = self.hand_drawn_mask_layer1

            if m1.shape != combined_mask.shape:
                m1 = cv2.resize(m1, (combined_mask.shape[1], combined_mask.shape[0]),
                                cv2.INTER_LINEAR)

            max_dev = self.neutral_mask_value
            if self.neutral_mask_value > 0.5:
                max_dev = 1.0 - self.neutral_mask_value
            if max_dev < 1e-6:
                max_dev = 0.5

            alpha = np.abs(m1 - self.neutral_mask_value) / max_dev
            alpha = np.clip(alpha, 0.0, 1.0)

            combined_mask = combined_mask * (1.0 - alpha) + m1 * alpha

        self.wrinkle_mask = np.clip(combined_mask, 0.0, 1.0)

    def load_micronormal(self, file_path):
        if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None: raise ValueError(f"Could not load image: {file_path}")
        self.micronormal_path = file_path
        self.micronormal_img = img
        if img.ndim >= 3:
            if img.shape[2] == 3:
                r_mn, g_mn, b_mn = cv2.split(img)
            elif img.shape[2] == 4:
                r_mn, g_mn, b_mn, a_mn = cv2.split(img)
            else:
                raise ValueError(f"Unsupported channel count: {img.shape[2]}")
            self.micronormal_x = (b_mn.astype(np.float32) / 255.0) * 2.0 - 1.0
            self.micronormal_y = (g_mn.astype(np.float32) / 255.0) * 2.0 - 1.0
        else:
            raise ValueError("Micronormal map must be a color image")
        return img

    def process_micronormal(self):
        if self.micronormal_x is None or self.micronormal_y is None or not self.use_micronormal: return
        if self.normal_x is None or self.normal_y is None: return
        main_h, main_w = self.normal_x.shape[:2]
        micro_h, micro_w = self.micronormal_x.shape[:2]
        scale_factor = 1.0 / self.micronormal_tile_size
        new_micro_w = int(micro_w * scale_factor)
        new_micro_h = int(micro_h * scale_factor)
        if new_micro_w <= 0 or new_micro_h <= 0: return
        resized_micro_x = cv2.resize(self.micronormal_x, (new_micro_w, new_micro_h), cv2.INTER_LINEAR)
        resized_micro_y = cv2.resize(self.micronormal_y, (new_micro_w, new_micro_h), cv2.INTER_LINEAR)
        num_tiles_h = int(np.ceil(main_h / new_micro_h)) if new_micro_h > 0 else 1
        num_tiles_w = int(np.ceil(main_w / new_micro_w)) if new_micro_w > 0 else 1
        tile_x = np.tile(resized_micro_x, (num_tiles_h, num_tiles_w))
        tile_y = np.tile(resized_micro_y, (num_tiles_h, num_tiles_w))
        self.tiled_micronormal_x = tile_x[:main_h, :main_w]
        self.tiled_micronormal_y = tile_y[:main_h, :main_w]

    def reduce_wrinkles(self):
        if self.normal_x is None or self.normal_y is None or self.wrinkle_mask is None:
            if self.normal_x is not None and self.normal_y is not None:
                self.processed_normal_x = self.normal_x.copy()
                self.processed_normal_y = self.normal_y.copy()
                proc_xy_sq = np.clip(self.processed_normal_x ** 2 + self.processed_normal_y ** 2, 0, 1)
                proc_norm_z = np.sqrt(1 - proc_xy_sq)
                self.processed_normal = cv2.merge([
                    ((self.processed_normal_x + 1) / 2 * 255).astype(np.uint8),
                    ((self.processed_normal_y + 1) / 2 * 255).astype(np.uint8),
                    (proc_norm_z * 255).astype(np.uint8)
                ])
            else:
                self.processed_normal_x = None; self.processed_normal_y = None; self.processed_normal = None
            return self.processed_normal

        normal_x_img = ((self.normal_x + 1.0) / 2.0 * 255.0).astype(np.uint8)
        normal_y_img = ((self.normal_y + 1.0) / 2.0 * 255.0).astype(np.uint8)

        if not self.detect_wrinkles_enabled:
            self.processed_normal_x = self.normal_x.copy()
            self.processed_normal_y = self.normal_y.copy()
            blended_xy_sq = np.clip(self.processed_normal_x ** 2 + self.processed_normal_y ** 2, 0, 1)
            blended_z = np.sqrt(1 - blended_xy_sq)
            length = np.sqrt(self.processed_normal_x ** 2 + self.processed_normal_y ** 2 + blended_z ** 2)
            length = np.maximum(length, 1e-6)
            self.processed_normal_x = self.processed_normal_x / length
            self.processed_normal_y = self.processed_normal_y / length
            processed_normal_z = blended_z / length
        else:
            smoothed_x = normal_x_img.copy()
            smoothed_y = normal_y_img.copy()

            d = max(9, int(15 * self.smoothing_strength)); d = d + 1 if d % 2 == 0 else d
            sigma_color = 75 + 50 * self.smoothing_strength
            sigma_space = 75 + 50 * self.smoothing_strength
            smoothed_x = cv2.bilateralFilter(smoothed_x, d, sigma_color, sigma_space)
            smoothed_y = cv2.bilateralFilter(smoothed_y, d, sigma_color, sigma_space)

            radius = max(4, int(8 * self.smoothing_strength))
            eps = 0.1 * (1 - self.smoothing_strength)
            smoothed_x = cv2.edgePreservingFilter(smoothed_x, flags=cv2.RECURS_FILTER, sigma_s=radius * 10, sigma_r=0.1 + eps)
            smoothed_y = cv2.edgePreservingFilter(smoothed_y, flags=cv2.RECURS_FILTER, sigma_s=radius * 10, sigma_r=0.1 + eps)

            if self.smoothing_strength > 0.7:
                iterations = int(3 * (self.smoothing_strength - 0.7) / 0.3)
                for _ in range(iterations):
                    smoothed_x = cv2.bilateralFilter(smoothed_x, 7, 30, 30)
                    smoothed_y = cv2.bilateralFilter(smoothed_y, 7, 30, 30)

            kernel_size_gauss = max(3, int(7 * self.smoothing_strength)); kernel_size_gauss = kernel_size_gauss + 1 if kernel_size_gauss % 2 == 0 else kernel_size_gauss
            smoothed_x = cv2.GaussianBlur(smoothed_x, (kernel_size_gauss, kernel_size_gauss), 0)
            smoothed_y = cv2.GaussianBlur(smoothed_y, (kernel_size_gauss, kernel_size_gauss), 0)

            smoothed_x_norm = (smoothed_x.astype(np.float32) / 255.0) * 2.0 - 1.0
            smoothed_y_norm = (smoothed_y.astype(np.float32) / 255.0) * 2.0 - 1.0

            blend_mask_for_smoothing = np.power(self.wrinkle_mask, 0.5) * self.smoothing_strength
            blend_mask_for_smoothing = np.clip(blend_mask_for_smoothing, 0, 1)

            blended_x = self.normal_x * (1 - blend_mask_for_smoothing) + smoothed_x_norm * blend_mask_for_smoothing
            blended_y = self.normal_y * (1 - blend_mask_for_smoothing) + smoothed_y_norm * blend_mask_for_smoothing

            blended_xy_sq = np.clip(blended_x ** 2 + blended_y ** 2, 0, 1)
            blended_z = np.sqrt(1 - blended_xy_sq)
            length = np.sqrt(blended_x ** 2 + blended_y ** 2 + blended_z ** 2); length = np.maximum(length, 1e-6)
            self.processed_normal_x = blended_x / length
            self.processed_normal_y = blended_y / length
            processed_normal_z = blended_z / length

        if hasattr(self, 'normal_intensity') and self.normal_intensity != 1.0:
            base_intensity_mask = np.ones_like(self.processed_normal_x)
            if self.normal_mask_mode == "inverse" and self.wrinkle_mask is not None:
                base_intensity_mask = 1.0 - self.wrinkle_mask
            elif self.normal_mask_mode == "direct" and self.wrinkle_mask is not None:
                base_intensity_mask = self.wrinkle_mask.copy()

            final_intensity_mask = base_intensity_mask
            if self.use_normal_custom_mask and self.normal_custom_mask is not None:
                custom_n_mask = self.normal_custom_mask
                if custom_n_mask.shape != base_intensity_mask.shape:
                    custom_n_mask = cv2.resize(custom_n_mask,
                                               (base_intensity_mask.shape[1], base_intensity_mask.shape[0]),
                                               cv2.INTER_LINEAR)
                final_intensity_mask = base_intensity_mask * custom_n_mask

            self.processed_normal_x = self.processed_normal_x * (1.0 - final_intensity_mask) + \
                                      self.processed_normal_x * self.normal_intensity * final_intensity_mask
            self.processed_normal_y = self.processed_normal_y * (1.0 - final_intensity_mask) + \
                                      self.processed_normal_y * self.normal_intensity * final_intensity_mask

            xy_sq_int = np.clip(self.processed_normal_x ** 2 + self.processed_normal_y ** 2, 0, 1)
            processed_normal_z = np.sqrt(1 - xy_sq_int)
            len_int = np.sqrt(self.processed_normal_x ** 2 + self.processed_normal_y ** 2 + processed_normal_z ** 2); len_int = np.maximum(len_int, 1e-6)
            self.processed_normal_x /= len_int; self.processed_normal_y /= len_int; processed_normal_z /= len_int

        if self.use_micronormal and self.tiled_micronormal_x is not None and self.tiled_micronormal_y is not None:
            base_micro_mask = np.ones_like(self.processed_normal_x)
            if self.micronormal_mask_mode == "inverse" and self.wrinkle_mask is not None:
                base_micro_mask = 1.0 - self.wrinkle_mask
            elif self.micronormal_mask_mode == "direct" and self.wrinkle_mask is not None:
                base_micro_mask = self.wrinkle_mask.copy()

            final_micro_mask = base_micro_mask
            if self.custom_micronormal_mask is not None:
                custom_m_mask = self.custom_micronormal_mask
                if custom_m_mask.shape != base_micro_mask.shape:
                    custom_m_mask = cv2.resize(custom_m_mask, (base_micro_mask.shape[1], base_micro_mask.shape[0]), cv2.INTER_LINEAR)
                final_micro_mask = base_micro_mask * custom_m_mask

            micro_xy_sq = np.clip(self.tiled_micronormal_x ** 2 + self.tiled_micronormal_y ** 2, 0, 1)
            micro_z = np.sqrt(1 - micro_xy_sq)
            blend_factor = final_micro_mask * self.micronormal_strength

            rot_micro_x = self.tiled_micronormal_x * processed_normal_z + self.processed_normal_x * micro_z
            rot_micro_y = self.tiled_micronormal_y * processed_normal_z + self.processed_normal_y * micro_z
            rot_micro_z = processed_normal_z * micro_z - (self.processed_normal_x * self.tiled_micronormal_x + self.processed_normal_y * self.tiled_micronormal_y)

            final_x = self.processed_normal_x * (1 - blend_factor) + rot_micro_x * blend_factor
            final_y = self.processed_normal_y * (1 - blend_factor) + rot_micro_y * blend_factor
            final_z = processed_normal_z * (1 - blend_factor) + rot_micro_z * blend_factor

            final_len = np.sqrt(final_x ** 2 + final_y ** 2 + final_z ** 2); final_len = np.maximum(final_len, 1e-6)
            self.processed_normal_x = final_x / final_len
            self.processed_normal_y = final_y / final_len
            processed_normal_z = final_z / final_len

        proc_norm_x_vis = ((self.processed_normal_x + 1.0) / 2.0 * 255.0).astype(np.uint8)
        proc_norm_y_vis = ((self.processed_normal_y + 1.0) / 2.0 * 255.0).astype(np.uint8)
        proc_norm_z_vis = (processed_normal_z * 255.0).astype(np.uint8)
        self.processed_normal = cv2.merge([proc_norm_x_vis, proc_norm_y_vis, proc_norm_z_vis])
        return self.processed_normal

    def combine_channels(self):
        if self.processed_normal_x is None or self.processed_normal_y is None or \
           self.adjusted_roughness_map is None or self.adjusted_specular_map is None:
            shape = self.original_img.shape if self.original_img is not None else (256, 256, 4)
            self.processed_img = np.zeros(shape, dtype=np.uint8)
            if shape[2] == 4: self.processed_img[:,:,3] = 255
            return self.processed_img

        nnrs_norm_x = ((self.processed_normal_x + 1.0) / 2.0 * 255.0).astype(np.uint8)
        nnrs_norm_y = ((self.processed_normal_y + 1.0) / 2.0 * 255.0).astype(np.uint8)
        self.processed_img = cv2.merge([
            self.adjusted_roughness_map,
            nnrs_norm_y,
            nnrs_norm_x,
            self.adjusted_specular_map
        ])
        return self.processed_img

    def save_textures(self, output_dir, base_name, save_separate=False):
        output_dir_path = Path(output_dir); output_dir_path.mkdir(parents=True, exist_ok=True)
        if self.processed_img is None: return None
        output_path = output_dir_path / f"{base_name}_processed.png"
        cv2.imwrite(str(output_path), self.processed_img)
        if save_separate:
            if self.processed_normal is not None:
                b, g, r = cv2.split(self.processed_normal)
                cv2.imwrite(str(output_dir_path / f"{base_name}_normal_processed_visual.png"), cv2.merge([r,g,b]))
            if self.adjusted_roughness_map is not None:
                cv2.imwrite(str(output_dir_path / f"{base_name}_roughness_adjusted.png"), self.adjusted_roughness_map)
            if self.adjusted_specular_map is not None:
                cv2.imwrite(str(output_dir_path / f"{base_name}_specular_adjusted.png"), self.adjusted_specular_map)
            if self.debug and self.wrinkle_mask is not None:
                cv2.imwrite(str(output_dir_path / f"{base_name}_wrinkle_mask_combined.png"), (self.wrinkle_mask * 255).astype(np.uint8))
        return str(output_path)

    def process(self, input_path, output_dir=None, base_name=None, save_separate=False):
        if output_dir is None: output_dir = os.path.dirname(input_path)
        if base_name is None: base_name = Path(input_path).stem

        self.load_texture(input_path)


        if self.use_roughness_custom_mask and self.roughness_custom_mask_path and self.roughness_custom_mask is None:
            try:
                mask_img = cv2.imread(self.roughness_custom_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None: self.roughness_custom_mask = mask_img
            except Exception as e:
                print(f"Warning: Process - Could not load roughness custom mask: {e}")

        if self.use_specular_custom_mask and self.specular_custom_mask_path and self.specular_custom_mask is None:
            try:
                mask_img = cv2.imread(self.specular_custom_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None: self.specular_custom_mask = mask_img
            except Exception as e:
                print(f"Warning: Process - Could not load specular custom mask: {e}")

        self.split_channels()
        self.adjust_roughness_specular()
        self.detect_wrinkles()
        if self.use_micronormal and self.micronormal_img is not None: self.process_micronormal()
        self.reduce_wrinkles()
        self.combine_channels()
        output_path = self.save_textures(output_dir, base_name, save_separate=save_separate)
        return self.processed_img, output_path

class BatchProcessor:
    def __init__(self):
        self.files = []
        self.results = []
        self.is_processing = False
        self.current_file_index = 0
        self.callback = None
        self.cancelled = False
        self.mask_template_layer1 = None

    def set_files(self, file_paths):
        self.files = file_paths
        self.results = []
        self.current_file_index = 0

    def set_callback(self, callback):
        self.callback = callback

    def cancel(self):
        self.cancelled = True

    def process(self, processor_params, output_dir):
        self.is_processing = True
        self.cancelled = False
        total_files = len(self.files)
        self.results = []
        save_separate_batch = processor_params.pop('save_separate', False)

        VALID_NNRS_INIT_PARAMS = [
            "wrinkle_threshold", "smoothing_strength", "debug",
            "roughness_brightness", "roughness_contrast",
            "specular_brightness", "specular_contrast",
            "use_micronormal", "micronormal_strength",
            "micronormal_tile_size", "micronormal_mask_mode",
            "detect_wrinkles_enabled",
            "normal_intensity", "normal_mask_mode",
            "use_normal_custom_mask",
            "enable_auto_mask_blur", "auto_mask_blur_strength",
            "use_roughness_custom_mask", "roughness_custom_mask_path",
            "use_specular_custom_mask", "specular_custom_mask_path"
        ]

        for i, file_path in enumerate(self.files):
            if self.cancelled:
                self.results.append({'file': file_path, 'status': 'cancelled', 'error': 'User cancelled'})
                if self.callback: self.callback('error', i, total_files, file_path, 'User cancelled')
                continue
            self.current_file_index = i
            try:
                nnrs_init_args = {k: v for k, v in processor_params.items() if k in VALID_NNRS_INIT_PARAMS}
                processor = NNRSTextureProcessor(**nnrs_init_args)

                if 'normal_custom_mask_path' in processor_params:
                    processor.normal_custom_mask_path = processor_params['normal_custom_mask_path']
                else:
                    processor.normal_custom_mask_path = None

                if processor.use_micronormal and 'micronormal_path' in processor_params and processor_params[
                    'micronormal_path']:
                    try:
                        processor.load_micronormal(processor_params['micronormal_path'])
                    except Exception as e:
                        if self.callback: self.callback('warning', i, total_files, file_path,
                                                        f"Micronormal loading error: {e}")

                if processor.use_micronormal and 'custom_micronormal_mask_path' in processor_params and \
                        processor_params['custom_micronormal_mask_path']:
                    try:
                        mask_img = cv2.imread(processor_params['custom_micronormal_mask_path'], cv2.IMREAD_GRAYSCALE)
                        if mask_img is not None:
                            processor.custom_micronormal_mask = mask_img.astype(np.float32) / 255.0
                    except Exception as e:
                        if self.callback: self.callback('warning', i, total_files, file_path,
                                                        f"Custom micronormal mask loading error: {e}")

                if processor_params.get('use_wrinkle_custom_mask', False):
                    processor.use_wrinkle_custom_mask = True
                else:
                    processor.use_wrinkle_custom_mask = False

                if processor.use_wrinkle_custom_mask and processor_params.get('wrinkle_custom_mask_path'):
                    try:
                        wrinkle_mask_path_val = processor_params['wrinkle_custom_mask_path']
                        mask_img = cv2.imread(wrinkle_mask_path_val, cv2.IMREAD_GRAYSCALE)
                        if mask_img is not None:
                            processor.wrinkle_custom_mask = mask_img.astype(np.float32) / 255.0
                            processor.wrinkle_custom_mask_mode = processor_params.get('wrinkle_custom_mask_mode',
                                                                                      "blend")
                            processor.wrinkle_custom_mask_path = wrinkle_mask_path_val
                        else:

                            processor.use_wrinkle_custom_mask = False
                            if self.callback: self.callback('warning', i, total_files, file_path,
                                                            f"Wrinkle custom mask file not found or unreadable: {wrinkle_mask_path_val}")
                    except Exception as e:
                        processor.use_wrinkle_custom_mask = False
                        if self.callback: self.callback('warning', i, total_files, file_path,
                                                        f"Error loading wrinkle custom mask: {e}")
                elif processor.use_wrinkle_custom_mask and not processor_params.get('wrinkle_custom_mask_path'):

                    processor.use_wrinkle_custom_mask = False
                    if self.callback: self.callback('warning', i, total_files, file_path,
                                                    "Use wrinkle custom mask checked, but no path provided.")

                if processor.use_normal_custom_mask and processor.normal_custom_mask_path:
                    try:
                        mask_img_normal = cv2.imread(processor.normal_custom_mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask_img_normal is not None:
                            processor.normal_custom_mask = mask_img_normal.astype(np.float32) / 255.0
                        else:
                            if self.callback: self.callback('warning', i, total_files, file_path,
                                                            f"Normal custom mask file not found or unreadable: {processor.normal_custom_mask_path}")
                    except Exception as e:
                        if self.callback: self.callback('warning', i, total_files, file_path,
                                                        f"Error loading normal custom mask: {e}")

                if processor.use_roughness_custom_mask and processor.roughness_custom_mask_path:
                    try:
                        mask_img = cv2.imread(processor.roughness_custom_mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask_img is not None:
                            processor.roughness_custom_mask = mask_img
                        else:
                            if self.callback: self.callback('warning', i, total_files, file_path,
                                                            f"Roughness custom mask not found/loaded: {processor.roughness_custom_mask_path}")
                    except Exception as e:
                        if self.callback: self.callback('warning', i, total_files, file_path,
                                                        f"Roughness custom mask loading error: {e}")

                if processor.use_specular_custom_mask and processor.specular_custom_mask_path:
                    try:
                        mask_img = cv2.imread(processor.specular_custom_mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask_img is not None:
                            processor.specular_custom_mask = mask_img
                        else:
                            if self.callback: self.callback('warning', i, total_files, file_path,
                                                            f"Specular custom mask not found/loaded: {processor.specular_custom_mask_path}")
                    except Exception as e:
                        if self.callback: self.callback('warning', i, total_files, file_path,
                                                        f"Specular custom mask loading error: {e}")

                if self.callback: self.callback('processing', i, total_files, file_path, None)

                if self.mask_template_layer1 is not None:
                    temp_original_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                    if temp_original_img is not None:
                        target_shape = temp_original_img.shape[:2]
                        resized_template_l1 = self.mask_template_layer1
                        if self.mask_template_layer1.shape != target_shape:
                            resized_template_l1 = cv2.resize(self.mask_template_layer1,
                                                             (target_shape[1], target_shape[0]),
                                                             interpolation=cv2.INTER_LINEAR)
                        processor.set_hand_drawn_mask_layer1(resized_template_l1,
                                                             processor_params.get('neutral_mask_value', 0.5))
                    else:
                        processor.set_hand_drawn_mask_layer1(self.mask_template_layer1,
                                                             processor_params.get('neutral_mask_value', 0.5))

                _, output_path = processor.process(
                    file_path,
                    output_dir=output_dir,
                    base_name=Path(file_path).stem,
                    save_separate=save_separate_batch
                )

                self.results.append({'file': file_path, 'status': 'success', 'output': output_path})
                if self.callback: self.callback('completed', i, total_files, file_path, output_path)
            except Exception as e:
                self.results.append({'file': file_path, 'status': 'error', 'error': str(e)})
                if self.callback: self.callback('error', i, total_files, file_path, str(e))

        self.is_processing = False
        if self.callback: self.callback('finished', total_files, total_files, None, self.results)
        return self.results

class NNRSTextureGUI:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Skin Texture Wrinkle Processing Tool v0.1")
        self.root.geometry("1280x960")
        self.root.minsize(1465, 850)
        self.setup_styles()
        self.processor = NNRSTextureProcessor()
        self.batch_processor = BatchProcessor()
        self.input_file = None
        self.output_dir = None
        self.base_name = None
        self.wrinkle_threshold = tk.DoubleVar(value=0.8)
        self.smoothing_strength = tk.DoubleVar(value=0.8)
        self.detect_wrinkles_enabled = tk.BooleanVar(value=True)
        self.enable_auto_mask_blur = tk.BooleanVar(value=True)
        self.auto_mask_blur_strength = tk.DoubleVar(value=0.2)
        self.auto_mask_layer0_gui = None
        self.hand_drawn_mask_layer1_gui = None
        self.neutral_mask_value = 0.5
        self.roughness_brightness = tk.DoubleVar(value=0.0)
        self.roughness_contrast = tk.DoubleVar(value=1.0)
        self.specular_brightness = tk.DoubleVar(value=0.0)
        self.specular_contrast = tk.DoubleVar(value=1.0)


        self.use_roughness_custom_mask = tk.BooleanVar(value=False)
        self.roughness_custom_mask_path_var = tk.StringVar(value="")
        self.roughness_custom_mask_gui = None

        self.use_specular_custom_mask = tk.BooleanVar(value=False)
        self.specular_custom_mask_path_var = tk.StringVar(value="")
        self.specular_custom_mask_gui = None

        self.save_separate = tk.BooleanVar(value=False)
        self.debug_mode = tk.BooleanVar(value=False)
        self.is_processing = False
        self.photo_references = {}
        self.fixed_display_width = 600
        self.fixed_display_height = 500
        self.is_drawing = False
        self.brush_size = tk.IntVar(value=20)
        self.brush_mode = tk.StringVar(value="add")
        self.brush_hardness = tk.DoubleVar(value=0.5)
        self.mask_has_edits = False
        self.last_x, self.last_y = None, None
        self.performance_tracker = PerformanceTracker()
        self.mask_editor = None
        self.batch_normal_intensity = tk.DoubleVar(value=1.0)
        self.batch_normal_mask_mode = tk.StringVar(value="none")
        self.batch_use_normal_custom_mask = tk.BooleanVar(value=False)
        self.batch_normal_custom_mask_path = None
        self.batch_normal_custom_mask = None
        self.batch_debug_mode = tk.BooleanVar(value=False)
        self.batch_files = []
        self.batch_output_dir = None
        self.batch_wrinkle_threshold = tk.DoubleVar(value=0.8)
        self.batch_smoothing_strength = tk.DoubleVar(value=0.8)
        self.batch_detect_wrinkles_enabled = tk.BooleanVar(value=True)
        self.batch_enable_auto_mask_blur = tk.BooleanVar(value=False)
        self.batch_auto_mask_blur_strength = tk.DoubleVar(value=0.2)
        self.batch_roughness_brightness = tk.DoubleVar(value=0.0)
        self.batch_roughness_contrast = tk.DoubleVar(value=1.0)
        self.batch_specular_brightness = tk.DoubleVar(value=0.0)
        self.batch_specular_contrast = tk.DoubleVar(value=1.0)


        self.batch_use_roughness_custom_mask = tk.BooleanVar(value=False)
        self.batch_roughness_custom_mask_path_var = tk.StringVar(value="")

        self.batch_use_specular_custom_mask = tk.BooleanVar(value=False)
        self.batch_specular_custom_mask_path_var = tk.StringVar(value="")

        self.batch_save_separate = tk.BooleanVar(value=False)
        self.use_mask_template_for_batch = tk.BooleanVar(value=False)
        self.batch_progress_queue = queue.Queue()
        self.configure_matplotlib()
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_styles(self):
        style = ttk.Style()
        self.colors = {
            'primary': '#3498db', 'secondary': '#2ecc71', 'accent': '#e74c3c',
            'bg_light': '#f5f7fa', 'bg_medium': '#ebeef2', 'bg_dark': '#d6dce4',
            'text_dark': '#2c3e50', 'text_medium': '#7f8c8d', 'text_light': '#bdc3c7',
            'border': '#cbd2d9'
        }
        default_font = ('Segoe UI', 9) if os.name == 'nt' else ('Helvetica', 9)
        heading_font = ('Segoe UI', 10, 'bold') if os.name == 'nt' else ('Helvetica', 10, 'bold')
        small_font = ('Segoe UI', 8) if os.name == 'nt' else ('Helvetica', 8)

        style.configure('.', font=default_font, background=self.colors['bg_light'], foreground=self.colors['text_dark'],
                        troughcolor=self.colors['bg_medium'], selectbackground=self.colors['primary'],
                        selectforeground='white', fieldbackground='white')
        style.configure('TNotebook', background=self.colors['bg_light'], tabmargins=[0, 4, 0, 0])
        style.configure('TNotebook.Tab', font=default_font, background=self.colors['bg_medium'],
                        foreground=self.colors['text_dark'], padding=[12, 4], borderwidth=0)
        style.map('TNotebook.Tab',
                  background=[('selected', self.colors['bg_light']), ('active', self.colors['bg_dark'])],
                  foreground=[('selected', self.colors['primary']), ('active', self.colors['text_dark'])])
        style.configure('Card.TLabelframe', background=self.colors['bg_light'], borderwidth=1, relief='solid')
        style.configure('Card.TLabelframe.Label', font=heading_font, background=self.colors['bg_light'],
                        foreground=self.colors['primary'])
        style.configure('TButton', font=default_font, background=self.colors['bg_medium'], padding=[8, 4])
        style.map('TButton', background=[('active', self.colors['bg_dark']), ('pressed', self.colors['bg_dark'])],
                  relief=[('pressed', 'sunken')])
        style.configure('Primary.TButton', background=self.colors['primary'], foreground=self.colors['text_dark'],
                        padding=[8, 4])
        style.map('Primary.TButton', background=[('active', '#2980b9'), ('pressed', '#1f6aa5')],
                  foreground=[('active', self.colors['text_dark']),
                              ('pressed', self.colors['text_dark'])])
        style.configure('Secondary.TButton', background=self.colors['secondary'], foreground=self.colors['text_dark'],
                        padding=[8, 4])
        style.map('Secondary.TButton', background=[('active', '#27ae60'), ('pressed', '#1e8449')],
                  foreground=[('active', self.colors['text_dark']),
                              ('pressed', self.colors['text_dark'])])
        style.configure('Danger.TButton', background=self.colors['accent'], foreground=self.colors['text_dark'],
                        padding=[8, 4])
        style.map('Danger.TButton', background=[('active', '#c0392b'), ('pressed', '#a5281b')],
                  foreground=[('active', self.colors['text_dark']),
                              ('pressed', self.colors['text_dark'])])
        style.configure('Title.TLabel', font=heading_font, foreground=self.colors['primary'],
                        background=self.colors['bg_light'])
        style.configure('TSeparator', background=self.colors['border'])
        style.configure('TProgressbar', thickness=6, background=self.colors['primary'])
        style.configure('TScale', sliderthickness=14, sliderlength=14, troughcolor=self.colors['bg_dark'])
        style.configure('TCheckbutton', background=self.colors['bg_light'], foreground=self.colors['text_dark'])
        style.configure('TRadiobutton', background=self.colors['bg_light'], foreground=self.colors['text_dark'])
        self.root.option_add('*TCombobox*Listbox.background', 'white')
        self.root.option_add('*TCombobox*Listbox.font', default_font)
        self.root.option_add('*TCombobox*Listbox.selectBackground', self.colors['primary'])
        self.root.option_add('*TCombobox*Listbox.selectForeground', 'white')

    def configure_matplotlib(self):
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                               'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        self.create_toolbar(main_frame)
        control_notebook = ttk.Notebook(main_frame, width=400)
        control_notebook.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        single_frame_tab = ttk.Frame(control_notebook)
        control_notebook.add(single_frame_tab, text="Single File")
        self.create_single_file_controls(single_frame_tab)
        batch_frame_tab = ttk.Frame(control_notebook)
        control_notebook.add(batch_frame_tab, text="Batch Process")
        self.create_batch_controls(batch_frame_tab)
        self.display_frame = ttk.Frame(main_frame, width=self.fixed_display_width, height=self.fixed_display_height)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.display_frame.pack_propagate(False)
        self.notebook = ttk.Notebook(self.display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.setup_image_tabs()
        self.setup_responsive_layout()

    def create_single_file_controls(self, parent_frame):
        canvas = tk.Canvas(parent_frame, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_canvas_configure_single(event):
            canvas.itemconfig(canvas_window_id, width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure_single)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel_single(event): canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel_single))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        file_frame = ttk.LabelFrame(scrollable_frame, text="File Selection", padding=10, style='Card.TLabelframe')
        file_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        bf_inner = ttk.Frame(file_frame)
        bf_inner.pack(fill=tk.X)
        self.file_entry = ttk.Entry(bf_inner)
        self.file_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        ttk.Button(bf_inner, text="Browse...", command=self.browse_file).pack(side=tk.RIGHT)

        save_frame = ttk.LabelFrame(scrollable_frame, text="Save Options", padding=10, style='Card.TLabelframe')
        save_frame.pack(fill=tk.X, padx=10, pady=5)
        sf_output_inner = ttk.Frame(save_frame)
        sf_output_inner.pack(fill=tk.X)
        self.output_entry = ttk.Entry(sf_output_inner)
        self.output_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        ttk.Button(sf_output_inner, text="Browse...", command=self.browse_output_dir).pack(side=tk.RIGHT)
        sf_name_inner = ttk.Frame(save_frame)
        sf_name_inner.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(sf_name_inner, text="Output File Name:").pack(side=tk.LEFT, padx=(0, 5))
        self.name_entry = ttk.Entry(sf_name_inner)
        self.name_entry.pack(fill=tk.X, expand=True)

        param_frame = ttk.LabelFrame(scrollable_frame, text="Processing Parameters", padding=10,
                                     style='Card.TLabelframe')
        param_frame.pack(fill=tk.X, padx=10, pady=5)
        self.create_parameter_controls(param_frame)

        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10, side=tk.BOTTOM)
        self.preview_btn = ttk.Button(button_frame, text="Preview", command=self.preview_textures,
                                      style='Primary.TButton')
        self.preview_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.process_btn = ttk.Button(button_frame, text="Process & Save", command=self.process_textures,
                                      style='Secondary.TButton')
        self.process_btn.pack(side=tk.RIGHT)
        self.view_normal_btn = ttk.Button(button_frame, text="View Normal", command=self.view_normal_map,
                                          state=tk.DISABLED)
        self.view_normal_btn.pack(side=tk.LEFT, padx=(0, 5))

        status_frame = ttk.Frame(scrollable_frame)
        status_frame.pack(fill=tk.X, padx=10, pady=(5, 0), side=tk.BOTTOM)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X)
        self.progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100).pack(fill=tk.X, pady=(2, 0))
        parent_frame.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))

    def create_parameter_controls(self, parent_frame):
        main_grid = ttk.Frame(parent_frame)
        main_grid.pack(fill=tk.X, pady=5)
        row = 0
        ttk.Checkbutton(main_grid, text="Enable Wrinkle Detection", variable=self.detect_wrinkles_enabled,
                        command=self.toggle_wrinkle_detection_and_children).grid(row=row, column=0, columnspan=2,
                                                                                 sticky=tk.W,
                                                                                 pady=2)
        row += 1
        ttk.Separator(main_grid, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=4)
        row += 1

        self.threshold_label = ttk.Label(main_grid, text="Wrinkle Detection Sensitivity:")
        self.threshold_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        self.threshold_label_val = ttk.Label(main_grid, text=f"{self.wrinkle_threshold.get():.2f}")
        self.threshold_label_val.grid(row=row, column=1, sticky=tk.E, pady=2)
        row += 1
        self.threshold_scale = ttk.Scale(main_grid, from_=0.05, to=5.0, variable=self.wrinkle_threshold,
                                         orient=tk.HORIZONTAL, command=lambda v: self.update_threshold_label(float(v)))
        self.threshold_scale.grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))
        row += 1

        self.strength_label = ttk.Label(main_grid, text="Normal Smoothing Strength:")
        self.strength_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        self.strength_label_val = ttk.Label(main_grid, text=f"{self.smoothing_strength.get():.2f}")
        self.strength_label_val.grid(row=row, column=1, sticky=tk.E, pady=2)
        row += 1
        self.strength_scale = ttk.Scale(main_grid, from_=0.1, to=1.0, variable=self.smoothing_strength,
                                        orient=tk.HORIZONTAL, command=lambda v: self.update_strength_label(float(v)))
        self.strength_scale.grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))
        row += 1

        self.auto_mask_blur_check = ttk.Checkbutton(main_grid, text="Enable Auto-Mask Blur (Layer 0)",
                                                    variable=self.enable_auto_mask_blur,
                                                    command=self.toggle_auto_mask_blur_controls)
        self.auto_mask_blur_check.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        self.auto_mask_blur_strength_label = ttk.Label(main_grid, text="Auto-Mask Blur Strength:")
        self.auto_mask_blur_strength_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        self.auto_mask_blur_strength_val_label = ttk.Label(main_grid, text=f"{self.auto_mask_blur_strength.get():.2f}")
        self.auto_mask_blur_strength_val_label.grid(row=row, column=1, sticky=tk.E, pady=2)
        row += 1
        self.auto_mask_blur_strength_scale = ttk.Scale(main_grid, from_=0.0, to=1.0,
                                                       variable=self.auto_mask_blur_strength,
                                                       orient=tk.HORIZONTAL,
                                                       command=lambda v: self.auto_mask_blur_strength_val_label.config(
                                                           text=f"{float(v):.2f}"))
        self.auto_mask_blur_strength_scale.grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))
        row += 1

        self.toggle_wrinkle_detection_and_children()
        self.toggle_auto_mask_blur_controls()

        ttk.Separator(main_grid, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=8)
        row += 1
        wrinkle_custom_mask_frame = ttk.LabelFrame(main_grid, text="Wrinkle Detection Custom Mask", padding=8,
                                                   style='Card.TLabelframe')
        wrinkle_custom_mask_frame.grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=5)
        row += 1
        self.use_wrinkle_custom_mask = tk.BooleanVar(value=False)
        ttk.Checkbutton(wrinkle_custom_mask_frame, text="Use Custom Wrinkle Mask",
                        variable=self.use_wrinkle_custom_mask,
                        command=self.toggle_wrinkle_custom_mask).pack(anchor=tk.W, pady=(0, 5))
        wcmf_file_frame = ttk.Frame(wrinkle_custom_mask_frame)
        wcmf_file_frame.pack(fill=tk.X, pady=2)
        ttk.Label(wcmf_file_frame, text="Mask File:").pack(side=tk.LEFT)
        self.wrinkle_custom_mask_entry = ttk.Entry(wcmf_file_frame, state=tk.DISABLED)
        self.wrinkle_custom_mask_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.browse_wrinkle_custom_mask_btn = ttk.Button(wcmf_file_frame, text="Browse...",
                                                         command=self.browse_wrinkle_custom_mask, state=tk.DISABLED)
        self.browse_wrinkle_custom_mask_btn.pack(side=tk.RIGHT)
        wcmf_mode_frame = ttk.Frame(wrinkle_custom_mask_frame)
        wcmf_mode_frame.pack(fill=tk.X, pady=2)
        self.wrinkle_custom_mask_mode = tk.StringVar(value="blend")
        ttk.Label(wcmf_mode_frame, text="Mask Mode:").pack(anchor=tk.W, pady=(0, 2))
        self.wrinkle_blend_radio = ttk.Radiobutton(wcmf_mode_frame, text="Blend",
                                                   variable=self.wrinkle_custom_mask_mode,
                                                   value="blend", state=tk.DISABLED)
        self.wrinkle_blend_radio.pack(anchor=tk.W, padx=(10, 0))
        self.wrinkle_replace_radio = ttk.Radiobutton(wcmf_mode_frame, text="Replace",
                                                     variable=self.wrinkle_custom_mask_mode, value="replace",
                                                     state=tk.DISABLED)
        self.wrinkle_replace_radio.pack(anchor=tk.W, padx=(10, 0))
        self.wrinkle_multiply_radio = ttk.Radiobutton(wcmf_mode_frame, text="Multiply",
                                                      variable=self.wrinkle_custom_mask_mode, value="multiply",
                                                      state=tk.DISABLED)
        self.wrinkle_multiply_radio.pack(anchor=tk.W, padx=(10, 0))
        self.wrinkle_subtract_radio = ttk.Radiobutton(wcmf_mode_frame, text="Subtract",
                                                      variable=self.wrinkle_custom_mask_mode, value="subtract",
                                                      state=tk.DISABLED)
        self.wrinkle_subtract_radio.pack(anchor=tk.W, padx=(10, 0))


        adjustment_frame = ttk.Frame(main_grid)
        adjustment_frame.grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=5)
        row += 1
        adjustment_frame.columnconfigure(0, weight=1)
        adjustment_frame.columnconfigure(1, weight=1)

        roughness_frame = ttk.LabelFrame(adjustment_frame, text="Roughness Adjustment", padding=8,
                                         style='Card.TLabelframe')
        roughness_frame.grid(row=0, column=0, sticky=tk.EW, padx=(0, 5))
        self.create_adjustment_controls(roughness_frame, "Brightness:", self.roughness_brightness, -1.0, 1.0,
                                        "Contrast:", self.roughness_contrast, 0.0, 3.0)

        ttk.Checkbutton(roughness_frame, text="Use Custom Mask",
                        variable=self.use_roughness_custom_mask,
                        command=self.toggle_roughness_custom_mask_single).pack(anchor=tk.W, pady=(5, 2))
        rcm_file_frame = ttk.Frame(roughness_frame)
        rcm_file_frame.pack(fill=tk.X, pady=2)
        rcm_file_frame.columnconfigure(0, weight=1)
        rcm_file_frame.columnconfigure(1, weight=0)

        self.roughness_custom_mask_entry_single = ttk.Entry(rcm_file_frame,
                                                            textvariable=self.roughness_custom_mask_path_var,
                                                            state=tk.DISABLED)
        self.roughness_custom_mask_entry_single.grid(row=0, column=0, sticky=tk.EW, padx=(0, 5))
        self.browse_roughness_custom_mask_btn_single = ttk.Button(rcm_file_frame, text="Browse...",
                                                                  command=lambda: self.browse_adj_custom_mask(
                                                                      "roughness_single"),
                                                                  state=tk.DISABLED)
        self.browse_roughness_custom_mask_btn_single.grid(row=0, column=1, sticky=tk.E)


        specular_frame = ttk.LabelFrame(adjustment_frame, text="Specular Adjustment", padding=8,
                                        style='Card.TLabelframe')
        specular_frame.grid(row=0, column=1, sticky=tk.EW, padx=(5, 0))
        self.create_adjustment_controls(specular_frame, "Brightness:", self.specular_brightness, -1.0, 1.0, "Contrast:",
                                        self.specular_contrast, 0.0, 3.0)

        ttk.Checkbutton(specular_frame, text="Use Custom Mask",
                        variable=self.use_specular_custom_mask,
                        command=self.toggle_specular_custom_mask_single).pack(anchor=tk.W, pady=(5, 2))
        scm_file_frame = ttk.Frame(specular_frame)
        scm_file_frame.pack(fill=tk.X, pady=2)
        scm_file_frame.columnconfigure(0, weight=1)
        scm_file_frame.columnconfigure(1, weight=0)

        self.specular_custom_mask_entry_single = ttk.Entry(scm_file_frame,
                                                           textvariable=self.specular_custom_mask_path_var,
                                                           state=tk.DISABLED)
        self.specular_custom_mask_entry_single.grid(row=0, column=0, sticky=tk.EW, padx=(0, 5))
        self.browse_specular_custom_mask_btn_single = ttk.Button(scm_file_frame, text="Browse...",
                                                                 command=lambda: self.browse_adj_custom_mask(
                                                                     "specular_single"),
                                                                 state=tk.DISABLED)
        self.browse_specular_custom_mask_btn_single.grid(row=0, column=1, sticky=tk.E)

        options_frame = ttk.Frame(main_grid)
        options_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=8)
        row += 1
        ttk.Checkbutton(options_frame, text="Save Separate Textures", variable=self.save_separate).pack(anchor=tk.W)
        ttk.Checkbutton(options_frame, text="Debug Mode (Show Extra Masks)", variable=self.debug_mode).pack(anchor=tk.W,
                                                                                                            pady=(2, 0))

        ttk.Separator(main_grid, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=8)
        row += 1
        normal_intensity_frame = ttk.LabelFrame(main_grid, text="Normal Intensity Adjustment", padding=8,
                                                style='Card.TLabelframe')
        normal_intensity_frame.grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=5)
        row += 1
        ni_frame = ttk.Frame(normal_intensity_frame)
        ni_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ni_frame, text="Intensity:").pack(side=tk.LEFT)
        self.normal_intensity = tk.DoubleVar(value=1.0)
        self.ni_val_label = ttk.Label(ni_frame, text=f"{self.normal_intensity.get():.2f}")
        self.ni_val_label.pack(side=tk.RIGHT)
        ttk.Scale(normal_intensity_frame, from_=0.0, to=2.0, variable=self.normal_intensity,
                  command=lambda v: self.ni_val_label.config(text=f"{float(v):.2f}")).pack(fill=tk.X, pady=(0, 5))
        normal_mask_options_frame = ttk.Frame(normal_intensity_frame)
        normal_mask_options_frame.pack(fill=tk.X, pady=2)
        ttk.Label(normal_mask_options_frame, text="Mask Mode:").pack(anchor=tk.W, pady=(0, 2))
        self.normal_mask_mode = tk.StringVar(value="none")
        ttk.Radiobutton(normal_mask_options_frame, text="No Mask", variable=self.normal_mask_mode,
                        value="none").pack(anchor=tk.W, padx=(10, 0))
        ttk.Radiobutton(normal_mask_options_frame, text="Use Inverse Mask", variable=self.normal_mask_mode,
                        value="inverse").pack(anchor=tk.W, padx=(10, 0))
        ttk.Radiobutton(normal_mask_options_frame, text="Use Direct Mask", variable=self.normal_mask_mode,
                        value="direct").pack(anchor=tk.W, padx=(10, 0))
        self.use_normal_custom_mask = tk.BooleanVar(value=False)
        ttk.Checkbutton(normal_mask_options_frame, text="Use Custom Mask", variable=self.use_normal_custom_mask,
                        command=self.toggle_normal_custom_mask).pack(anchor=tk.W, pady=(5, 0))
        normal_custom_mask_file_frame = ttk.Frame(normal_intensity_frame)
        normal_custom_mask_file_frame.pack(fill=tk.X, pady=2)
        ttk.Label(normal_custom_mask_file_frame, text="Custom Mask:").pack(side=tk.LEFT)
        self.normal_custom_mask_entry = ttk.Entry(normal_custom_mask_file_frame, state=tk.DISABLED)
        self.normal_custom_mask_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.browse_normal_custom_mask_btn = ttk.Button(normal_custom_mask_file_frame, text="Browse...",
                                                        command=self.browse_normal_custom_mask, state=tk.DISABLED)
        self.browse_normal_custom_mask_btn.pack(side=tk.RIGHT)

        ttk.Separator(main_grid, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=8)
        row += 1
        micronormal_frame = ttk.LabelFrame(main_grid, text="Micronormal Overlay", padding=8, style='Card.TLabelframe')
        micronormal_frame.grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=5)
        row += 1
        self.use_micronormal = tk.BooleanVar(value=False)
        ttk.Checkbutton(micronormal_frame, text="Enable Micronormal Overlay", variable=self.use_micronormal,
                        command=self.toggle_micronormal_controls).pack(anchor=tk.W, pady=(0, 5))
        micro_file_frame = ttk.Frame(micronormal_frame)
        micro_file_frame.pack(fill=tk.X, pady=2)
        ttk.Label(micro_file_frame, text="Micronormal Map:").pack(side=tk.LEFT)
        self.micronormal_entry = ttk.Entry(micro_file_frame, state=tk.DISABLED)
        self.micronormal_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.browse_micronormal_btn = ttk.Button(micro_file_frame, text="Browse...", command=self.browse_micronormal,
                                                 state=tk.DISABLED)
        self.browse_micronormal_btn.pack(side=tk.RIGHT)
        self.micronormal_controls_frame = ttk.Frame(micronormal_frame)
        self.micronormal_controls_frame.pack(fill=tk.X, pady=5)
        micro_strength_frame = ttk.Frame(self.micronormal_controls_frame)
        micro_strength_frame.pack(fill=tk.X, pady=2)
        ttk.Label(micro_strength_frame, text="Blend Strength:").pack(side=tk.LEFT)
        self.micronormal_strength = tk.DoubleVar(value=0.5)
        self.micro_strength_label = ttk.Label(micro_strength_frame, text=f"{self.micronormal_strength.get():.2f}")
        self.micro_strength_label.pack(side=tk.RIGHT)
        self.micro_strength_scale = ttk.Scale(self.micronormal_controls_frame, from_=0.0, to=1.0,
                                              variable=self.micronormal_strength,
                                              command=lambda v: self.micro_strength_label.config(
                                                  text=f"{float(v):.2f}"), state=tk.DISABLED)
        self.micro_strength_scale.pack(fill=tk.X, pady=(0, 5))
        micro_tile_frame = ttk.Frame(self.micronormal_controls_frame)
        micro_tile_frame.pack(fill=tk.X, pady=2)
        ttk.Label(micro_tile_frame, text="Tile Size:").pack(side=tk.LEFT)
        self.micronormal_tile_size = tk.DoubleVar(value=1.0)
        self.micro_tile_label = ttk.Label(micro_tile_frame, text=f"{self.micronormal_tile_size.get():.2f}x")
        self.micro_tile_label.pack(side=tk.RIGHT)
        self.micro_tile_scale = ttk.Scale(self.micronormal_controls_frame, from_=0.1, to=10.0,
                                          variable=self.micronormal_tile_size,
                                          command=lambda v: self.micro_tile_label.config(text=f"{float(v):.2f}x"),
                                          state=tk.DISABLED)
        self.micro_tile_scale.pack(fill=tk.X, pady=(0, 5))
        micro_mask_options_frame = ttk.Frame(self.micronormal_controls_frame)
        micro_mask_options_frame.pack(fill=tk.X, pady=2)
        ttk.Label(micro_mask_options_frame, text="Mask Mode:").pack(anchor=tk.W, pady=(0, 2))
        self.micronormal_mask_mode = tk.StringVar(value="inverse")
        self.none_mask_radio = ttk.Radiobutton(micro_mask_options_frame, text="No Mask",
                                               variable=self.micronormal_mask_mode, value="none", state=tk.DISABLED)
        self.none_mask_radio.pack(anchor=tk.W, padx=(10, 0))
        self.inverse_mask_radio = ttk.Radiobutton(micro_mask_options_frame, text="Use Inverse Mask",
                                                  variable=self.micronormal_mask_mode, value="inverse",
                                                  state=tk.DISABLED)
        self.inverse_mask_radio.pack(anchor=tk.W, padx=(10, 0))
        self.direct_mask_radio = ttk.Radiobutton(micro_mask_options_frame, text="Use Direct Mask",
                                                 variable=self.micronormal_mask_mode, value="direct",
                                                 state=tk.DISABLED)
        self.direct_mask_radio.pack(anchor=tk.W, padx=(10, 0))
        self.use_custom_mask = tk.BooleanVar(value=False)
        self.use_custom_mask_check = ttk.Checkbutton(micro_mask_options_frame, text="Use Custom Mask",
                                                     variable=self.use_custom_mask, command=self.toggle_custom_mask,
                                                     state=tk.DISABLED)
        self.use_custom_mask_check.pack(anchor=tk.W, pady=(5, 0))
        custom_mask_file_frame = ttk.Frame(self.micronormal_controls_frame)
        custom_mask_file_frame.pack(fill=tk.X, pady=2)
        ttk.Label(custom_mask_file_frame, text="Custom Mask:").pack(side=tk.LEFT)
        self.custom_mask_entry = ttk.Entry(custom_mask_file_frame, state=tk.DISABLED)
        self.custom_mask_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.browse_custom_mask_btn = ttk.Button(custom_mask_file_frame, text="Browse...",
                                                 command=self.browse_custom_mask, state=tk.DISABLED)
        self.browse_custom_mask_btn.pack(side=tk.RIGHT)

        main_grid.columnconfigure(0, weight=1)
        main_grid.columnconfigure(1, weight=1)
        self.toggle_wrinkle_custom_mask()
        self.toggle_normal_custom_mask()
        self.toggle_micronormal_controls()
        self.toggle_roughness_custom_mask_single()
        self.toggle_specular_custom_mask_single()

    def toggle_roughness_custom_mask_single(self):
        state = tk.NORMAL if self.use_roughness_custom_mask.get() else tk.DISABLED
        self.roughness_custom_mask_entry_single.config(state=state)
        self.browse_roughness_custom_mask_btn_single.config(state=state)
        if not self.use_roughness_custom_mask.get():
            self.roughness_custom_mask_path_var.set("")
            self.roughness_custom_mask_gui = None

    def toggle_specular_custom_mask_single(self):
        state = tk.NORMAL if self.use_specular_custom_mask.get() else tk.DISABLED
        self.specular_custom_mask_entry_single.config(state=state)
        self.browse_specular_custom_mask_btn_single.config(state=state)
        if not self.use_specular_custom_mask.get():
            self.specular_custom_mask_path_var.set("")
            self.specular_custom_mask_gui = None

    def browse_adj_custom_mask(self, mask_type_mode):
        title = "Select Custom Mask for "
        entry_var = None
        loaded_mask_attr = None

        if mask_type_mode == "roughness_single":
            title += "Roughness (Single File)"
            entry_var = self.roughness_custom_mask_path_var

        elif mask_type_mode == "specular_single":
            title += "Specular (Single File)"
            entry_var = self.specular_custom_mask_path_var

        elif mask_type_mode == "roughness_batch":
            title += "Roughness (Batch)"
            entry_var = self.batch_roughness_custom_mask_path_var
        elif mask_type_mode == "specular_batch":
            title += "Specular (Batch)"
            entry_var = self.batch_specular_custom_mask_path_var
        else:
            return

        file_path = filedialog.askopenfilename(title=title,
                                               filetypes=[("Image files", "*.png *.jpg *.jpeg *.tga *.bmp"),
                                                          ("All files", "*.*")])
        if file_path and entry_var:
            entry_var.set(file_path)
            if "_single" in mask_type_mode:
                try:
                    mask_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if mask_img is None: raise ValueError("Could not load mask image")

                    if mask_type_mode == "roughness_single":
                        self.roughness_custom_mask_gui = mask_img
                    elif mask_type_mode == "specular_single":
                        self.specular_custom_mask_gui = mask_img
                    self.status_var.set(f"Loaded {mask_type_mode.replace('_', ' ')} mask: {Path(file_path).name}")
                except Exception as e:
                    messagebox.showerror("Mask Loading Error", f"Could not load mask for {mask_type_mode}: {e}")
                    entry_var.set("")
                    if mask_type_mode == "roughness_single":
                        self.roughness_custom_mask_gui = None
                    elif mask_type_mode == "specular_single":
                        self.specular_custom_mask_gui = None

    def create_adjustment_controls(self, parent, b_label, b_var, b_from, b_to, c_label, c_var, c_from, c_to):
        b_frame = ttk.Frame(parent)
        b_frame.pack(fill=tk.X, pady=2)
        ttk.Label(b_frame, text=b_label).pack(side=tk.LEFT)
        b_val_label = ttk.Label(b_frame, text=f"{b_var.get():.2f}", width=4)
        b_val_label.pack(side=tk.RIGHT)
        ttk.Scale(parent, from_=b_from, to=b_to, variable=b_var, orient=tk.HORIZONTAL,
                  command=lambda v: b_val_label.config(text=f"{float(v):.2f}")).pack(fill=tk.X, pady=(0, 4))
        c_frame = ttk.Frame(parent)
        c_frame.pack(fill=tk.X, pady=2)
        ttk.Label(c_frame, text=c_label).pack(side=tk.LEFT)
        c_val_label = ttk.Label(c_frame, text=f"{c_var.get():.2f}", width=4)
        c_val_label.pack(side=tk.RIGHT)
        ttk.Scale(parent, from_=c_from, to=c_to, variable=c_var, orient=tk.HORIZONTAL,
                  command=lambda v: c_val_label.config(text=f"{float(v):.2f}")).pack(fill=tk.X, pady=(0, 2))

    def setup_responsive_layout(self):
        def on_window_resize(event):
            if event.widget == self.root:
                window_width = event.width
                window_height = event.height
                if hasattr(self, 'display_frame') and hasattr(self, 'notebook'):
                    padding = 20
                    control_width = min(400, int(window_width * 0.35))
                    display_width = window_width - control_width - padding
                    display_height = window_height - padding - (
                        self.toolbar.winfo_height() if hasattr(self, 'toolbar') else 0)
                    self.display_frame.config(width=max(100, display_width), height=max(100, display_height))
                    current_tab = self.notebook.select()
                    if current_tab: self.notebook.event_generate("<<NotebookTabChanged>>")

        self.root.bind("<Configure>", on_window_resize)

    def create_batch_controls(self, parent_frame):
        canvas = tk.Canvas(parent_frame, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window_id_batch = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        parent_frame.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window_id_batch, width=e.width))
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel_batch(event): canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel_batch))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        batch_file_frame = ttk.LabelFrame(scrollable_frame, text="Batch File Selection", padding=10,
                                          style='Card.TLabelframe')
        batch_file_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        file_buttons_frame = ttk.Frame(batch_file_frame)
        file_buttons_frame.pack(fill=tk.X, pady=5)
        ttk.Button(file_buttons_frame, text="Select Files...", command=self.select_batch_files).pack(side=tk.LEFT,
                                                                                                     padx=(0, 5))
        ttk.Button(file_buttons_frame, text="Select Folder...", command=self.select_batch_folder).pack(side=tk.LEFT)
        ttk.Button(file_buttons_frame, text="Clear", command=self.clear_batch_files, style='Danger.TButton').pack(
            side=tk.RIGHT)
        list_frame = ttk.Frame(batch_file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.batch_files_listbox = tk.Listbox(list_frame, height=6, selectmode=tk.EXTENDED, activestyle='none')
        self.batch_files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_sb_y = ttk.Scrollbar(list_frame, orient="vertical", command=self.batch_files_listbox.yview)
        list_sb_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.batch_files_listbox.configure(yscrollcommand=list_sb_y.set)
        list_sb_x = ttk.Scrollbar(list_frame, orient="horizontal", command=self.batch_files_listbox.xview)
        list_sb_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.batch_files_listbox.configure(xscrollcommand=list_sb_x.set)
        self.batch_file_count_label = ttk.Label(batch_file_frame, text="0 files selected")
        self.batch_file_count_label.pack(anchor=tk.W)

        batch_output_frame = ttk.LabelFrame(scrollable_frame, text="Batch Output Settings", padding=10,
                                            style='Card.TLabelframe')
        batch_output_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(batch_output_frame, text="Output Directory:").pack(anchor=tk.W)
        bof_inner = ttk.Frame(batch_output_frame)
        bof_inner.pack(fill=tk.X, pady=5)
        self.batch_output_entry = ttk.Entry(bof_inner)
        self.batch_output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(bof_inner, text="Browse...", command=self.browse_batch_output_dir).pack(side=tk.RIGHT, padx=(5, 0))

        batch_param_frame = ttk.LabelFrame(scrollable_frame, text="Batch Parameters", padding=10,
                                           style='Card.TLabelframe')
        batch_param_frame.pack(fill=tk.X, padx=10, pady=5)
        batch_main_grid = ttk.Frame(batch_param_frame)
        batch_main_grid.pack(fill=tk.X, pady=5)
        row_b = 0

        self.inherit_single_params = tk.BooleanVar(value=False)
        self.inherit_params_check = ttk.Checkbutton(batch_main_grid, text="Inherit Single File Parameters",
                                                    variable=self.inherit_single_params,
                                                    command=self.toggle_inherit_params)
        self.inherit_params_check.grid(row=row_b, column=0, columnspan=2, sticky=tk.W, pady=2)
        row_b += 1
        ttk.Separator(batch_main_grid, orient=tk.HORIZONTAL).grid(row=row_b, column=0, columnspan=2, sticky=tk.EW,
                                                                  pady=4)
        row_b += 1

        self.batch_detect_wrinkles_check = ttk.Checkbutton(batch_main_grid, text="Enable Wrinkle Detection",
                                                           variable=self.batch_detect_wrinkles_enabled,
                                                           command=self.toggle_batch_wrinkle_detection_and_children)
        self.batch_detect_wrinkles_check.grid(row=row_b, column=0, columnspan=2, sticky=tk.W, pady=2)
        row_b += 1
        ttk.Separator(batch_main_grid, orient=tk.HORIZONTAL).grid(row=row_b, column=0, columnspan=2, sticky=tk.EW,
                                                                  pady=4)
        row_b += 1
        self.batch_threshold_label = ttk.Label(batch_main_grid, text="Wrinkle Detection Sensitivity:")
        self.batch_threshold_label.grid(row=row_b, column=0, sticky=tk.W, pady=2)
        self.batch_threshold_label_val = ttk.Label(batch_main_grid, text=f"{self.batch_wrinkle_threshold.get():.2f}")
        self.batch_threshold_label_val.grid(row=row_b, column=1, sticky=tk.E, pady=2)
        row_b += 1
        self.batch_threshold_scale = ttk.Scale(batch_main_grid, from_=0.05, to=5.0,
                                               variable=self.batch_wrinkle_threshold, orient=tk.HORIZONTAL,
                                               command=lambda v: self.update_batch_threshold_label(float(v)))
        self.batch_threshold_scale.grid(row=row_b, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))
        row_b += 1
        self.batch_strength_label = ttk.Label(batch_main_grid, text="Normal Smoothing Strength:")
        self.batch_strength_label.grid(row=row_b, column=0, sticky=tk.W, pady=2)
        self.batch_strength_label_val = ttk.Label(batch_main_grid, text=f"{self.batch_smoothing_strength.get():.2f}")
        self.batch_strength_label_val.grid(row=row_b, column=1, sticky=tk.E, pady=2)
        row_b += 1
        self.batch_strength_scale = ttk.Scale(batch_main_grid, from_=0.1, to=1.0,
                                              variable=self.batch_smoothing_strength, orient=tk.HORIZONTAL,
                                              command=lambda v: self.update_batch_strength_label(float(v)))
        self.batch_strength_scale.grid(row=row_b, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))
        row_b += 1

        self.batch_auto_mask_blur_check = ttk.Checkbutton(batch_main_grid, text="Enable Auto-Mask Blur (Layer 0)",
                                                          variable=self.batch_enable_auto_mask_blur,
                                                          command=self.toggle_batch_auto_mask_blur_controls)
        self.batch_auto_mask_blur_check.grid(row=row_b, column=0, columnspan=2, sticky=tk.W, pady=2)
        row_b += 1

        self.batch_auto_mask_blur_strength_label = ttk.Label(batch_main_grid, text="Auto-Mask Blur Strength:")
        self.batch_auto_mask_blur_strength_label.grid(row=row_b, column=0, sticky=tk.W, pady=2)
        self.batch_auto_mask_blur_strength_val_label = ttk.Label(batch_main_grid,
                                                                 text=f"{self.batch_auto_mask_blur_strength.get():.2f}")
        self.batch_auto_mask_blur_strength_val_label.grid(row=row_b, column=1, sticky=tk.E, pady=2)
        row_b += 1
        self.batch_auto_mask_blur_strength_scale = ttk.Scale(batch_main_grid, from_=0.0, to=1.0,
                                                             variable=self.batch_auto_mask_blur_strength,
                                                             orient=tk.HORIZONTAL,
                                                             command=lambda
                                                                 v: self.batch_auto_mask_blur_strength_val_label.config(
                                                                 text=f"{float(v):.2f}"))
        self.batch_auto_mask_blur_strength_scale.grid(row=row_b, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))
        row_b += 1

        ttk.Separator(batch_main_grid, orient=tk.HORIZONTAL).grid(row=row_b, column=0, columnspan=2, sticky=tk.EW,
                                                                  pady=8)
        row_b += 1
        batch_wrinkle_custom_mask_frame = ttk.LabelFrame(batch_main_grid, text="Wrinkle Detection Custom Mask (Batch)",
                                                         padding=8,
                                                         style='Card.TLabelframe')
        batch_wrinkle_custom_mask_frame.grid(row=row_b, column=0, columnspan=2, sticky=tk.EW, pady=5)
        row_b += 1
        self.batch_use_wrinkle_custom_mask = tk.BooleanVar(value=False)
        self.batch_use_wrinkle_custom_mask_check = ttk.Checkbutton(batch_wrinkle_custom_mask_frame,
                                                                   text="Use Custom Wrinkle Mask",
                                                                   variable=self.batch_use_wrinkle_custom_mask,
                                                                   command=self.toggle_batch_wrinkle_custom_mask)
        self.batch_use_wrinkle_custom_mask_check.pack(anchor=tk.W, pady=(0, 5))

        bwcmf_file_frame = ttk.Frame(batch_wrinkle_custom_mask_frame)
        bwcmf_file_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bwcmf_file_frame, text="Mask File:").pack(side=tk.LEFT)
        self.batch_wrinkle_custom_mask_entry = ttk.Entry(bwcmf_file_frame, state=tk.DISABLED)
        self.batch_wrinkle_custom_mask_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.batch_browse_wrinkle_custom_mask_btn = ttk.Button(bwcmf_file_frame, text="Browse...",
                                                               command=self.browse_batch_wrinkle_custom_mask,
                                                               state=tk.DISABLED)
        self.batch_browse_wrinkle_custom_mask_btn.pack(side=tk.RIGHT)
        bwcmf_mode_frame = ttk.Frame(batch_wrinkle_custom_mask_frame)
        bwcmf_mode_frame.pack(fill=tk.X, pady=2)
        self.batch_wrinkle_custom_mask_mode = tk.StringVar(value="blend")
        ttk.Label(bwcmf_mode_frame, text="Mask Mode:").pack(anchor=tk.W, pady=(0, 2))
        self.batch_wrinkle_blend_radio = ttk.Radiobutton(bwcmf_mode_frame, text="Blend",
                                                         variable=self.batch_wrinkle_custom_mask_mode, value="blend",
                                                         state=tk.DISABLED)
        self.batch_wrinkle_blend_radio.pack(anchor=tk.W, padx=(10, 0))
        self.batch_wrinkle_replace_radio = ttk.Radiobutton(bwcmf_mode_frame, text="Replace",
                                                           variable=self.batch_wrinkle_custom_mask_mode,
                                                           value="replace", state=tk.DISABLED)
        self.batch_wrinkle_replace_radio.pack(anchor=tk.W, padx=(10, 0))
        self.batch_wrinkle_multiply_radio = ttk.Radiobutton(bwcmf_mode_frame, text="Multiply",
                                                            variable=self.batch_wrinkle_custom_mask_mode,
                                                            value="multiply", state=tk.DISABLED)
        self.batch_wrinkle_multiply_radio.pack(anchor=tk.W, padx=(10, 0))
        self.batch_wrinkle_subtract_radio = ttk.Radiobutton(bwcmf_mode_frame, text="Subtract",
                                                            variable=self.batch_wrinkle_custom_mask_mode,
                                                            value="subtract", state=tk.DISABLED)
        self.batch_wrinkle_subtract_radio.pack(anchor=tk.W, padx=(10, 0))

        batch_adjustment_frame = ttk.Frame(batch_main_grid)
        batch_adjustment_frame.grid(row=row_b, column=0, columnspan=2, sticky=tk.EW, pady=5)
        row_b += 1
        batch_adjustment_frame.columnconfigure(0, weight=1)
        batch_adjustment_frame.columnconfigure(1, weight=1)

        self.batch_roughness_frame = ttk.LabelFrame(batch_adjustment_frame, text="Roughness Adjustment", padding=8,
                                                    style='Card.TLabelframe')
        self.batch_roughness_frame.grid(row=0, column=0, sticky=tk.EW, padx=(0, 5))
        self.create_adjustment_controls(self.batch_roughness_frame, "Brightness:", self.batch_roughness_brightness,
                                        -1.0, 1.0,
                                        "Contrast:", self.batch_roughness_contrast, 0.0, 3.0)

        self.batch_use_roughness_custom_mask_check = ttk.Checkbutton(self.batch_roughness_frame, text="Use Custom Mask",
                                                                    variable=self.batch_use_roughness_custom_mask,
                                                                    command=self.toggle_roughness_custom_mask_batch)
        self.batch_use_roughness_custom_mask_check.pack(anchor=tk.W, pady=(5,2))
        brcm_file_frame = ttk.Frame(self.batch_roughness_frame)
        brcm_file_frame.pack(fill=tk.X, pady=2)
        brcm_file_frame.columnconfigure(0, weight=1)
        brcm_file_frame.columnconfigure(1, weight=0)

        self.batch_roughness_custom_mask_entry = ttk.Entry(brcm_file_frame,
                                                           textvariable=self.batch_roughness_custom_mask_path_var,
                                                           state=tk.DISABLED)
        self.batch_roughness_custom_mask_entry.grid(row=0, column=0, sticky=tk.EW, padx=(0, 5))
        self.batch_browse_roughness_custom_mask_btn = ttk.Button(brcm_file_frame, text="Browse...",
                                                                 command=lambda: self.browse_adj_custom_mask(
                                                                     "roughness_batch"),
                                                                 state=tk.DISABLED)
        self.batch_browse_roughness_custom_mask_btn.grid(row=0, column=1, sticky=tk.E)


        self.batch_specular_frame = ttk.LabelFrame(batch_adjustment_frame, text="Specular Adjustment", padding=8,
                                                   style='Card.TLabelframe')
        self.batch_specular_frame.grid(row=0, column=1, sticky=tk.EW, padx=(5, 0))
        self.create_adjustment_controls(self.batch_specular_frame, "Brightness:", self.batch_specular_brightness, -1.0,
                                        1.0,
                                        "Contrast:", self.batch_specular_contrast, 0.0, 3.0)

        self.batch_use_specular_custom_mask_check = ttk.Checkbutton(self.batch_specular_frame, text="Use Custom Mask",
                                                                   variable=self.batch_use_specular_custom_mask,
                                                                   command=self.toggle_specular_custom_mask_batch)
        self.batch_use_specular_custom_mask_check.pack(anchor=tk.W, pady=(5,2))
        bscm_file_frame = ttk.Frame(self.batch_specular_frame)
        bscm_file_frame.pack(fill=tk.X, pady=2)
        bscm_file_frame.columnconfigure(0, weight=1)
        bscm_file_frame.columnconfigure(1, weight=0)

        self.batch_specular_custom_mask_entry = ttk.Entry(bscm_file_frame,
                                                          textvariable=self.batch_specular_custom_mask_path_var,
                                                          state=tk.DISABLED)
        self.batch_specular_custom_mask_entry.grid(row=0, column=0, sticky=tk.EW, padx=(0, 5))
        self.batch_browse_specular_custom_mask_btn = ttk.Button(bscm_file_frame, text="Browse...",
                                                                command=lambda: self.browse_adj_custom_mask(
                                                                    "specular_batch"),
                                                                state=tk.DISABLED)
        self.batch_browse_specular_custom_mask_btn.grid(row=0, column=1, sticky=tk.E)

        batch_options_frame = ttk.Frame(batch_main_grid)
        batch_options_frame.grid(row=row_b, column=0, columnspan=2, sticky=tk.W, pady=8)
        row_b += 1

        self.batch_save_separate_check = ttk.Checkbutton(batch_options_frame, text="Save Separate Textures",
                                                         variable=self.batch_save_separate)
        self.batch_save_separate_check.pack(anchor=tk.W)

        ttk.Checkbutton(batch_options_frame, text="Use Current Edited Mask (Layer 1) as Template",
                        variable=self.use_mask_template_for_batch).pack(anchor=tk.W, pady=(2, 0))
        self.batch_debug_mode_check = ttk.Checkbutton(batch_options_frame, text="Debug Mode (Save Extra Masks)",
                                                      variable=self.batch_debug_mode)
        self.batch_debug_mode_check.pack(anchor=tk.W, pady=(2,0))


        ttk.Separator(batch_main_grid, orient=tk.HORIZONTAL).grid(row=row_b, column=0, columnspan=2, sticky=tk.EW,
                                                                  pady=8)
        row_b += 1
        batch_normal_intensity_frame = ttk.LabelFrame(batch_main_grid, text="Normal Intensity Adjustment (Batch)",
                                                      padding=8,
                                                      style='Card.TLabelframe')
        batch_normal_intensity_frame.grid(row=row_b, column=0, columnspan=2, sticky=tk.EW, pady=5)
        row_b += 1
        bni_frame = ttk.Frame(batch_normal_intensity_frame)
        bni_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bni_frame, text="Intensity:").pack(side=tk.LEFT)
        self.bni_val_label = ttk.Label(bni_frame, text=f"{self.batch_normal_intensity.get():.2f}")
        self.bni_val_label.pack(side=tk.RIGHT)
        self.batch_normal_intensity_scale = ttk.Scale(batch_normal_intensity_frame, from_=0.0, to=2.0,
                                                      variable=self.batch_normal_intensity,
                                                      command=lambda v: self.bni_val_label.config(
                                                          text=f"{float(v):.2f}"))
        self.batch_normal_intensity_scale.pack(fill=tk.X, pady=(0, 5))

        self.bnormal_mask_options_frame = ttk.Frame(batch_normal_intensity_frame)
        self.bnormal_mask_options_frame.pack(fill=tk.X, pady=2)
        ttk.Label(self.bnormal_mask_options_frame, text="Mask Mode:").pack(anchor=tk.W, pady=(0, 2))
        ttk.Radiobutton(self.bnormal_mask_options_frame, text="No Mask", variable=self.batch_normal_mask_mode,
                        value="none").pack(anchor=tk.W, padx=(10, 0))
        ttk.Radiobutton(self.bnormal_mask_options_frame, text="Use Inverse Mask", variable=self.batch_normal_mask_mode,
                        value="inverse").pack(anchor=tk.W, padx=(10, 0))
        ttk.Radiobutton(self.bnormal_mask_options_frame, text="Use Direct Mask", variable=self.batch_normal_mask_mode,
                        value="direct").pack(anchor=tk.W, padx=(10, 0))
        self.batch_use_normal_custom_mask_check = ttk.Checkbutton(self.bnormal_mask_options_frame,
                                                                  text="Use Custom Mask",
                                                                  variable=self.batch_use_normal_custom_mask,
                                                                  command=self.toggle_batch_normal_custom_mask)
        self.batch_use_normal_custom_mask_check.pack(anchor=tk.W, pady=(5, 0))

        bnormal_custom_mask_file_frame = ttk.Frame(batch_normal_intensity_frame)
        bnormal_custom_mask_file_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bnormal_custom_mask_file_frame, text="Custom Mask:").pack(side=tk.LEFT)
        self.batch_normal_custom_mask_entry = ttk.Entry(bnormal_custom_mask_file_frame, state=tk.DISABLED)
        self.batch_normal_custom_mask_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.batch_browse_normal_custom_mask_btn = ttk.Button(bnormal_custom_mask_file_frame, text="Browse...",
                                                              command=self.browse_batch_normal_custom_mask,
                                                              state=tk.DISABLED)
        self.batch_browse_normal_custom_mask_btn.pack(side=tk.RIGHT)

        ttk.Separator(batch_main_grid, orient=tk.HORIZONTAL).grid(row=row_b, column=0, columnspan=2, sticky=tk.EW,
                                                                  pady=8)
        row_b += 1
        batch_micronormal_frame = ttk.LabelFrame(batch_main_grid, text="Micronormal Overlay (Batch)", padding=8,
                                                 style='Card.TLabelframe')
        batch_micronormal_frame.grid(row=row_b, column=0, columnspan=2, sticky=tk.EW, pady=5)
        row_b += 1
        self.batch_use_micronormal = tk.BooleanVar(value=False)
        self.batch_use_micronormal_check = ttk.Checkbutton(batch_micronormal_frame, text="Enable Micronormal Overlay",
                                                           variable=self.batch_use_micronormal,
                                                           command=self.toggle_batch_micronormal_controls)
        self.batch_use_micronormal_check.pack(anchor=tk.W, pady=(0, 5))

        bmicro_file_frame = ttk.Frame(batch_micronormal_frame)
        bmicro_file_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bmicro_file_frame, text="Micronormal Map:").pack(side=tk.LEFT)
        self.batch_micronormal_entry = ttk.Entry(bmicro_file_frame, state=tk.DISABLED)
        self.batch_micronormal_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.batch_browse_micronormal_btn = ttk.Button(bmicro_file_frame, text="Browse...",
                                                       command=self.browse_batch_micronormal, state=tk.DISABLED)
        self.batch_browse_micronormal_btn.pack(side=tk.RIGHT)

        self.batch_micronormal_controls_frame = ttk.Frame(batch_micronormal_frame)
        self.batch_micronormal_controls_frame.pack(fill=tk.X, pady=5)

        bmicro_strength_frame = ttk.Frame(self.batch_micronormal_controls_frame)
        bmicro_strength_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bmicro_strength_frame, text="Blend Strength:").pack(side=tk.LEFT)
        self.batch_micronormal_strength = tk.DoubleVar(value=0.5)
        self.batch_micro_strength_label = ttk.Label(bmicro_strength_frame,
                                                    text=f"{self.batch_micronormal_strength.get():.2f}")
        self.batch_micro_strength_label.pack(side=tk.RIGHT)
        self.batch_micro_strength_scale = ttk.Scale(self.batch_micronormal_controls_frame, from_=0.0, to=1.0,
                                                    variable=self.batch_micronormal_strength,
                                                    command=lambda v: self.batch_micro_strength_label.config(
                                                        text=f"{float(v):.2f}"), state=tk.DISABLED)
        self.batch_micro_strength_scale.pack(fill=tk.X, pady=(0, 5))

        bmicro_tile_frame = ttk.Frame(self.batch_micronormal_controls_frame)
        bmicro_tile_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bmicro_tile_frame, text="Tile Size:").pack(side=tk.LEFT)
        self.batch_micronormal_tile_size = tk.DoubleVar(value=1.0)
        self.batch_micro_tile_label = ttk.Label(bmicro_tile_frame,
                                                text=f"{self.batch_micronormal_tile_size.get():.2f}x")
        self.batch_micro_tile_label.pack(side=tk.RIGHT)
        self.batch_micro_tile_scale = ttk.Scale(self.batch_micronormal_controls_frame, from_=0.1, to=10.0,
                                                variable=self.batch_micronormal_tile_size,
                                                command=lambda v: self.batch_micro_tile_label.config(
                                                    text=f"{float(v):.2f}x"), state=tk.DISABLED)
        self.batch_micro_tile_scale.pack(fill=tk.X, pady=(0, 5))

        self.bmicro_mask_options_frame = ttk.Frame(self.batch_micronormal_controls_frame)
        self.bmicro_mask_options_frame.pack(fill=tk.X, pady=2)
        ttk.Label(self.bmicro_mask_options_frame, text="Mask Mode:").pack(anchor=tk.W, pady=(0, 2))
        self.batch_micronormal_mask_mode = tk.StringVar(value="inverse")
        self.batch_none_mask_radio = ttk.Radiobutton(self.bmicro_mask_options_frame, text="No Mask",
                                                     variable=self.batch_micronormal_mask_mode, value="none",
                                                     state=tk.DISABLED)
        self.batch_none_mask_radio.pack(anchor=tk.W, padx=(10, 0))
        self.batch_inverse_mask_radio = ttk.Radiobutton(self.bmicro_mask_options_frame, text="Use Inverse Mask",
                                                        variable=self.batch_micronormal_mask_mode, value="inverse",
                                                        state=tk.DISABLED)
        self.batch_inverse_mask_radio.pack(anchor=tk.W, padx=(10, 0))
        self.batch_direct_mask_radio = ttk.Radiobutton(self.bmicro_mask_options_frame, text="Use Direct Mask",
                                                       variable=self.batch_micronormal_mask_mode, value="direct",
                                                       state=tk.DISABLED)
        self.batch_direct_mask_radio.pack(anchor=tk.W, padx=(10, 0))

        self.batch_use_custom_mask = tk.BooleanVar(value=False)
        self.batch_use_custom_mask_check = ttk.Checkbutton(self.bmicro_mask_options_frame, text="Use Custom Mask",
                                                           variable=self.batch_use_custom_mask,
                                                           command=self.toggle_batch_custom_mask, state=tk.DISABLED)
        self.batch_use_custom_mask_check.pack(anchor=tk.W, pady=(5, 0))

        bcustom_mask_file_frame = ttk.Frame(self.batch_micronormal_controls_frame)
        bcustom_mask_file_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bcustom_mask_file_frame, text="Custom Mask:").pack(side=tk.LEFT)
        self.batch_custom_mask_entry = ttk.Entry(bcustom_mask_file_frame, state=tk.DISABLED)
        self.batch_custom_mask_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.batch_browse_custom_mask_btn = ttk.Button(bcustom_mask_file_frame, text="Browse...",
                                                       command=self.browse_batch_custom_mask, state=tk.DISABLED)
        self.batch_browse_custom_mask_btn.pack(side=tk.RIGHT)

        batch_main_grid.columnconfigure(0, weight=1)
        batch_main_grid.columnconfigure(1, weight=1)

        batch_control_frame = ttk.Frame(scrollable_frame)
        batch_control_frame.pack(fill=tk.X, padx=10, pady=10, side=tk.BOTTOM)
        self.batch_start_btn = ttk.Button(batch_control_frame, text="Start Batch Process",
                                          command=self.start_batch_processing,
                                          style='Primary.TButton')
        self.batch_start_btn.pack(side=tk.LEFT)
        self.batch_cancel_btn = ttk.Button(batch_control_frame, text="Cancel", command=self.cancel_batch_processing,
                                           state=tk.DISABLED, style='Danger.TButton')
        self.batch_cancel_btn.pack(side=tk.LEFT, padx=(5, 0))

        batch_progress_log_frame = ttk.Frame(scrollable_frame)
        batch_progress_log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5, side=tk.BOTTOM)
        batch_progress_frame = ttk.LabelFrame(batch_progress_log_frame, text="Batch Progress", padding=10,
                                              style='Card.TLabelframe')
        batch_progress_frame.pack(fill=tk.X)
        ttk.Label(batch_progress_frame, text="Overall Progress:").pack(anchor=tk.W)
        self.batch_progress_var = tk.DoubleVar(value=0.0)
        self.batch_progress_bar = ttk.Progressbar(batch_progress_frame, variable=self.batch_progress_var, maximum=100)
        self.batch_progress_bar.pack(fill=tk.X, pady=(5, 10))
        self.batch_current_file_label = ttk.Label(batch_progress_frame, text="Waiting to start...")
        self.batch_current_file_label.pack(anchor=tk.W)
        log_frame = ttk.LabelFrame(batch_progress_log_frame, text="Processing Log", padding=10,
                                   style='Card.TLabelframe')
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.batch_log_text = tk.Text(log_frame, height=8, wrap=tk.WORD, relief=tk.SUNKEN, borderwidth=1,
                                      state=tk.DISABLED, font=('Consolas', 8) if os.name == 'nt' else ('Monaco', 9))
        self.batch_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_sb_y = ttk.Scrollbar(log_frame, orient="vertical", command=self.batch_log_text.yview)
        log_sb_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.batch_log_text.configure(yscrollcommand=log_sb_y.set)
        parent_frame.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))

        self.toggle_inherit_params()

    def toggle_roughness_custom_mask_batch(self):
        is_enabled_overall = self.batch_use_roughness_custom_mask.get() and not self.inherit_single_params.get()
        state = tk.NORMAL if is_enabled_overall else tk.DISABLED
        if hasattr(self, 'batch_roughness_custom_mask_entry'):
            self.batch_roughness_custom_mask_entry.config(state=state)
            self.batch_browse_roughness_custom_mask_btn.config(state=state)
        if not self.batch_use_roughness_custom_mask.get() and hasattr(self, 'batch_roughness_custom_mask_path_var'):
            self.batch_roughness_custom_mask_path_var.set("")

    def toggle_specular_custom_mask_batch(self):
        is_enabled_overall = self.batch_use_specular_custom_mask.get() and not self.inherit_single_params.get()
        state = tk.NORMAL if is_enabled_overall else tk.DISABLED
        if hasattr(self, 'batch_specular_custom_mask_entry'):
            self.batch_specular_custom_mask_entry.config(state=state)
            self.batch_browse_specular_custom_mask_btn.config(state=state)
        if not self.batch_use_specular_custom_mask.get() and hasattr(self, 'batch_specular_custom_mask_path_var'):
            self.batch_specular_custom_mask_path_var.set("")

    def toggle_inherit_params(self):
        if self.inherit_single_params.get():
            self._copy_params_from_single_to_batch()
            self._set_batch_control_state(tk.DISABLED)
        else:
            self._set_batch_control_state(tk.NORMAL)

    def _set_batch_control_state(self, state):

        if hasattr(self, 'batch_detect_wrinkles_check'):
            self.batch_detect_wrinkles_check.config(state=state)
        self.toggle_batch_wrinkle_detection_and_children()

        if hasattr(self, 'batch_use_wrinkle_custom_mask_check'):
            self.batch_use_wrinkle_custom_mask_check.config(state=state)
        self.toggle_batch_wrinkle_custom_mask()

        def set_children_state(parent_widget, current_state):
            if parent_widget and hasattr(parent_widget, 'winfo_children'):
                for child in parent_widget.winfo_children():
                    if isinstance(child, (ttk.Scale, ttk.Label, ttk.Entry, ttk.Button)):
                        try: child.config(state=current_state)
                        except tk.TclError: pass
                    elif isinstance(child, ttk.Checkbutton):

                         try: child.config(state=current_state)
                         except tk.TclError: pass
                         if current_state == tk.NORMAL:
                             if child == self.batch_use_roughness_custom_mask_check: self.toggle_roughness_custom_mask_batch()
                             elif child == self.batch_use_specular_custom_mask_check: self.toggle_specular_custom_mask_batch()
                         else:
                             if child == self.batch_use_roughness_custom_mask_check: self.toggle_roughness_custom_mask_batch()
                             elif child == self.batch_use_specular_custom_mask_check: self.toggle_specular_custom_mask_batch()

                    elif isinstance(child, ttk.Frame):
                        set_children_state(child, current_state)

        if hasattr(self, 'batch_roughness_frame'):
            set_children_state(self.batch_roughness_frame, state)

            if hasattr(self, 'batch_use_roughness_custom_mask_check'):
                 self.batch_use_roughness_custom_mask_check.config(state=state)
                 self.toggle_roughness_custom_mask_batch()

        if hasattr(self, 'batch_specular_frame'):
            set_children_state(self.batch_specular_frame, state)
            if hasattr(self, 'batch_use_specular_custom_mask_check'):
                self.batch_use_specular_custom_mask_check.config(state=state)
                self.toggle_specular_custom_mask_batch()

        if hasattr(self, 'batch_normal_intensity_scale'): self.batch_normal_intensity_scale.config(state=state)
        if hasattr(self, 'bni_val_label'): self.bni_val_label.config(state=state)
        if hasattr(self, 'bnormal_mask_options_frame'):
            for widget in self.bnormal_mask_options_frame.winfo_children():
                if isinstance(widget, (ttk.Radiobutton, ttk.Checkbutton, ttk.Label)):
                    try: widget.config(state=state)
                    except tk.TclError: pass
        self.toggle_batch_normal_custom_mask()

        if hasattr(self, 'batch_use_micronormal_check'):
            self.batch_use_micronormal_check.config(state=state)
        self.toggle_batch_micronormal_controls()

        if hasattr(self, 'batch_save_separate_check'): self.batch_save_separate_check.config(state=state)
        if hasattr(self, 'batch_debug_mode_check'): self.batch_debug_mode_check.config(state=state)

        if hasattr(self, 'use_mask_template_for_batch_check') and isinstance(self.use_mask_template_for_batch_check, ttk.Checkbutton):
            self.use_mask_template_for_batch_check.config(state=state)

    def _copy_params_from_single_to_batch(self):
        self.batch_detect_wrinkles_enabled.set(self.detect_wrinkles_enabled.get())
        self.batch_wrinkle_threshold.set(self.wrinkle_threshold.get())
        self.batch_smoothing_strength.set(self.smoothing_strength.get())
        if hasattr(self, 'batch_threshold_label_val'): self.update_batch_threshold_label(self.wrinkle_threshold.get())
        if hasattr(self, 'batch_strength_label_val'): self.update_batch_strength_label(self.smoothing_strength.get())

        self.batch_enable_auto_mask_blur.set(self.enable_auto_mask_blur.get())
        self.batch_auto_mask_blur_strength.set(self.auto_mask_blur_strength.get())
        if hasattr(self, 'batch_auto_mask_blur_strength_val_label'):
            self.batch_auto_mask_blur_strength_val_label.config(text=f"{self.auto_mask_blur_strength.get():.2f}")

        self.batch_use_wrinkle_custom_mask.set(self.use_wrinkle_custom_mask.get())
        self.batch_wrinkle_custom_mask_mode.set(self.wrinkle_custom_mask_mode.get())
        if self.use_wrinkle_custom_mask.get() and hasattr(self, 'wrinkle_custom_mask_entry') and hasattr(self,
                                                                                                         'batch_wrinkle_custom_mask_entry'):
            path = self.wrinkle_custom_mask_entry.get()
            self.batch_wrinkle_custom_mask_entry.delete(0, tk.END)
            if path: self.batch_wrinkle_custom_mask_entry.insert(0, path)

        self.batch_roughness_brightness.set(self.roughness_brightness.get())
        self.batch_roughness_contrast.set(self.roughness_contrast.get())
        self.batch_specular_brightness.set(self.specular_brightness.get())
        self.batch_specular_contrast.set(self.specular_contrast.get())

        self.batch_use_roughness_custom_mask.set(self.use_roughness_custom_mask.get())
        self.batch_roughness_custom_mask_path_var.set(self.roughness_custom_mask_path_var.get())

        self.batch_use_specular_custom_mask.set(self.use_specular_custom_mask.get())
        self.batch_specular_custom_mask_path_var.set(self.specular_custom_mask_path_var.get())

        self.batch_normal_intensity.set(self.normal_intensity.get())
        self.batch_normal_mask_mode.set(self.normal_mask_mode.get())
        self.batch_use_normal_custom_mask.set(self.use_normal_custom_mask.get())
        if self.use_normal_custom_mask.get() and hasattr(self, 'normal_custom_mask_entry') and hasattr(self,
                                                                                                       'batch_normal_custom_mask_entry'):
            path = self.normal_custom_mask_entry.get()
            self.batch_normal_custom_mask_entry.delete(0, tk.END)
            if path: self.batch_normal_custom_mask_entry.insert(0, path)

        self.batch_use_micronormal.set(self.use_micronormal.get())
        self.batch_micronormal_strength.set(self.micronormal_strength.get())
        self.batch_micronormal_tile_size.set(self.micronormal_tile_size.get())
        self.batch_micronormal_mask_mode.set(self.micronormal_mask_mode.get())
        self.batch_use_custom_mask.set(self.use_custom_mask.get())

        if hasattr(self, 'micronormal_entry') and hasattr(self, 'batch_micronormal_entry'):
            path_mn = self.micronormal_entry.get()
            self.batch_micronormal_entry.delete(0, tk.END)
            if path_mn: self.batch_micronormal_entry.insert(0, path_mn)

        if hasattr(self, 'custom_mask_entry') and hasattr(self, 'batch_custom_mask_entry'):
            path_cmnm = self.custom_mask_entry.get()
            self.batch_custom_mask_entry.delete(0, tk.END)
            if path_cmnm: self.batch_custom_mask_entry.insert(0, path_cmnm)

        if hasattr(self, 'batch_micro_strength_label'): self.batch_micro_strength_label.config(
            text=f"{self.batch_micronormal_strength.get():.2f}")
        if hasattr(self, 'batch_micro_tile_label'): self.batch_micro_tile_label.config(
            text=f"{self.batch_micronormal_tile_size.get():.2f}x")
        if hasattr(self, 'bni_val_label'): self.bni_val_label.config(text=f"{self.batch_normal_intensity.get():.2f}")

        self.batch_debug_mode.set(self.debug_mode.get())
        self.batch_save_separate.set(self.save_separate.get())

    def setup_image_tabs(self):
        tab_configs = [
            ("Original Texture", "original_frame", "original_canvas", "Original NNRS Texture (RGBA)"),
            ("Normal Map", "normal_frame", "normal_canvas", "Original Normal Map (Visual RGB)"),
            ("Roughness", "roughness_frame_display", "roughness_canvas_display", "Roughness Map (Grayscale)"),
            ("Specular", "specular_frame_display", "specular_canvas_display", "Specular Map (Grayscale)"),
            ("Combined Mask", "mask_frame", "mask_canvas_widget", "Wrinkle Mask (Auto + Manual)"),
            ("Micronormal", "micronormal_frame", "micronormal_canvas", "Micronormal Map (RGB)"),
            ("Processed Texture", "processed_frame", "processed_canvas", "Processed NNRS Texture (RGBA)"),
            ("Processed Normal", "processed_normal_frame", "processed_normal_canvas", "Processed Normal (Visual RGB)"),
            ("Edit Manual Mask (Layer 1)", "mask_edit_frame", None, None)
        ]

        for name, frame_attr, canvas_attr, _ in tab_configs:
            frame = ttk.Frame(self.notebook)
            frame.pack_propagate(False)
            setattr(self, frame_attr, frame)
            self.notebook.add(frame, text=name)

            if canvas_attr and name == "Combined Mask":
                self.mask_fig = plt.Figure(figsize=(5, 4), dpi=100, facecolor=self.colors['bg_medium'])
                self.mask_fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
                mask_canvas_widget = FigureCanvasTkAgg(self.mask_fig, master=getattr(self, frame_attr))
                mask_canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                setattr(self, "mask_canvas_mpl", mask_canvas_widget)
            elif canvas_attr:
                canvas = tk.Canvas(frame, bg=self.colors['bg_dark'], highlightthickness=0)
                canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                setattr(self, canvas_attr, canvas)

        mask_edit_main_frame = getattr(self, "mask_edit_frame")
        mask_edit_main_frame.pack_propagate(False)
        self.mask_edit_container = ttk.Frame(mask_edit_main_frame)
        self.mask_edit_container.pack(fill=tk.BOTH, expand=True)
        self.mask_edit_controls = ttk.Frame(self.mask_edit_container)
        self.mask_edit_controls.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.create_mask_editor_controls(self.mask_edit_controls)
        self.mask_edit_canvas_frame = ttk.Frame(self.mask_edit_container)
        self.mask_edit_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.mask_edit_canvas = tk.Canvas(self.mask_edit_canvas_frame, bg=self.colors['bg_dark'], highlightthickness=0)
        self.mask_edit_canvas.pack(fill=tk.BOTH, expand=True)
        self.mask_edit_canvas.bind("<Button-1>", self.on_mask_edit_start)
        self.mask_edit_canvas.bind("<B1-Motion>", self.on_mask_edit_drag)
        self.mask_edit_canvas.bind("<ButtonRelease-1>", self.on_mask_edit_stop)
        self.mask_edit_canvas.bind("<MouseWheel>", self._on_mousewheel_mask_edit)
        self.mask_edit_canvas.bind("<Enter>", self.on_mask_edit_canvas_enter)
        self.mask_edit_canvas.bind("<Leave>", self.on_mask_edit_canvas_leave)
        self.mask_edit_canvas.bind("<Motion>", self.on_mask_edit_canvas_motion)

    def on_mask_edit_canvas_enter(self, event):
        if self.notebook.select() == str(self.mask_edit_frame):
            self.update_brush_preview(event.x, event.y)

    def on_mask_edit_canvas_leave(self, event):
        self.remove_brush_preview()

    def on_mask_edit_canvas_motion(self, event):
        if self.notebook.select() == str(
                self.mask_edit_frame) and not self.is_drawing:
            self.update_brush_preview(event.x, event.y)

    def _on_mousewheel_mask_edit(self, event):
        if not self.is_drawing:
            current_size = self.brush_size.get()
            if event.delta > 0:
                new_size = min(current_size + 5, 150)
            else:
                new_size = max(current_size - 5, 5)
            self.brush_size.set(new_size)
            if hasattr(self, 'size_combobox'): self.size_combobox.set(new_size)
            if self.last_x is not None and self.last_y is not None:
                self.update_brush_preview(self.mask_edit_canvas.winfo_pointerx() - self.mask_edit_canvas.winfo_rootx(),
                                          self.mask_edit_canvas.winfo_pointery() - self.mask_edit_canvas.winfo_rooty())

    def on_tab_changed(self, event):
        if not hasattr(self.processor, 'original_img') or self.processor.original_img is None: return
        try:
            selected_tab_widget_name = self.notebook.select()
            selected_tab_widget = self.notebook.nametowidget(selected_tab_widget_name)
        except tk.TclError:
            return

        if selected_tab_widget == self.original_frame:
            self.display_image(self.processor.original_img, self.original_canvas, "Original NNRS Texture (RGBA)")
        elif selected_tab_widget == self.normal_frame:
            self.display_image(self.processor.normal_map, self.normal_canvas, "Original Normal Map (Visual RGB)")
        elif selected_tab_widget == self.roughness_frame_display:
            self.display_image(
                self.processor.adjusted_roughness_map if self.processor.adjusted_roughness_map is not None else self.processor.roughness_map,
                self.roughness_canvas_display, "Roughness Map (Grayscale)")
        elif selected_tab_widget == self.specular_frame_display:
            self.display_image(
                self.processor.adjusted_specular_map if self.processor.adjusted_specular_map is not None else self.processor.specular_map,
                self.specular_canvas_display, "Specular Map (Grayscale)")
        elif selected_tab_widget == self.mask_frame:
            if self.processor.wrinkle_mask is not None and self.processor.gradient_magnitude is not None:
                self.display_mask(self.processor.wrinkle_mask, self.processor.gradient_magnitude)
            else:
                self.clear_mpl_canvas(self.mask_fig, self.mask_canvas_mpl, "Combined mask data unavailable")
        elif selected_tab_widget == self.micronormal_frame:
            if self.processor.micronormal_img is not None:
                if len(self.processor.micronormal_img.shape) == 3 and self.processor.micronormal_img.shape[2] >= 3:
                    visual_micronormal = cv2.cvtColor(self.processor.micronormal_img[:, :, :3], cv2.COLOR_BGR2RGB)
                    self.display_image(visual_micronormal, self.micronormal_canvas, "Micronormal Map (RGB)")
                elif self.processor.micronormal_x is not None and self.processor.micronormal_y is not None:
                    x_vis_mn = ((self.processor.micronormal_x + 1) / 2 * 255).astype(np.uint8)
                    y_vis_mn = ((self.processor.micronormal_y + 1) / 2 * 255).astype(np.uint8)
                    xy_sq_mn = np.clip(self.processor.micronormal_x ** 2 + self.processor.micronormal_y ** 2, 0, 1)
                    z_mn = np.sqrt(1 - xy_sq_mn)
                    z_vis_mn = (z_mn * 255).astype(np.uint8)
                    visual_mn_rgb = cv2.merge([x_vis_mn, y_vis_mn, z_vis_mn])
                    self.display_image(visual_mn_rgb, self.micronormal_canvas, "Micronormal Map (Reconstructed RGB)")
            else:
                self.clear_canvas(self.micronormal_canvas, "Micronormal map not loaded")
        elif selected_tab_widget == self.processed_frame:
            self.display_image(self.processor.processed_img, self.processed_canvas, "Processed NNRS Texture (RGBA)")
        elif selected_tab_widget == self.processed_normal_frame:
            self.display_image(self.processor.processed_normal, self.processed_normal_canvas,
                               "Processed Normal (Visual RGB)")
        elif selected_tab_widget == self.mask_edit_frame:
            if self.hand_drawn_mask_layer1_gui is not None:
                if not hasattr(self, 'mask_editor') or self.mask_editor is None: self.init_mask_editor()
                self.mask_editor.init_from_mask(self.hand_drawn_mask_layer1_gui.copy(), self.neutral_mask_value)
                self.update_mask_editor_display_with_layers()
            else:
                self.clear_canvas(self.mask_edit_canvas, "Please preview first to initialize Layer 1")

    def clear_canvas(self, canvas, message="Data unavailable"):
        canvas.delete("all")
        canvas_width = canvas.winfo_width() if canvas.winfo_width() > 0 else self.fixed_display_width
        canvas_height = canvas.winfo_height() if canvas.winfo_height() > 0 else self.fixed_display_height
        canvas.create_text(canvas_width // 2, canvas_height // 2, text=message, fill="grey", font=("Arial", 10))

    def clear_mpl_canvas(self, fig, canvas_mpl, message="Data unavailable"):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center', color='grey', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        canvas_mpl.draw()

    def update_threshold_label(self, value):
        self.threshold_label_val.config(text=f"{value:.2f}")

    def update_strength_label(self, value):
        self.strength_label_val.config(text=f"{value:.2f}")

    def browse_file(self):
        file_path = filedialog.askopenfilename(title="Select NNRS Texture File",
                                               filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            self.input_file = file_path
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
            self.output_dir = os.path.dirname(file_path)
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, self.output_dir)
            self.base_name = Path(file_path).stem
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, self.base_name)
            self.status_var.set(f"Selected: {Path(file_path).name}")
            try:
                self.processor.load_texture(self.input_file)
                self.processor.split_channels()
                self.processor.adjust_roughness_specular()
                self._initialize_mask_layers()
                self.processor.set_hand_drawn_mask_layer1(self.hand_drawn_mask_layer1_gui, self.neutral_mask_value)
                self.processor.detect_wrinkles()
                self.auto_mask_layer0_gui = self.processor.auto_mask_layer0.copy() if self.processor.auto_mask_layer0 is not None else None
                self._update_preview_ui()
                self.notebook.select(self.original_frame)
                self.view_normal_btn.config(state=tk.NORMAL if self.processor.normal_map is not None else tk.DISABLED)
            except Exception as e:
                messagebox.showerror("Loading Error", f"Failed to load file: {e}")

    def _initialize_mask_layers(self):
        if self.processor.original_img is None: return
        target_shape = self.processor.original_img.shape[:2]
        if self.hand_drawn_mask_layer1_gui is None or self.hand_drawn_mask_layer1_gui.shape != target_shape:
            self.hand_drawn_mask_layer1_gui = np.full(target_shape, self.neutral_mask_value, dtype=np.float32)
        if self.mask_editor:
            self.mask_editor.init_from_mask(self.hand_drawn_mask_layer1_gui.copy(), self.neutral_mask_value)

    def browse_output_dir(self):
        output_dir_selected = filedialog.askdirectory(title="Select Output Directory")
        if output_dir_selected:
            self.output_dir = output_dir_selected
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, self.output_dir)

    def display_image(self, img_data, canvas, title=None):
        canvas.delete("all")
        if img_data is None:
            self.clear_canvas(canvas, f"{title.split('(')[0].strip() if title else 'Image'} data unavailable")
            return

        img_data_u8 = img_data
        if img_data.dtype != np.uint8:
            if np.max(img_data) <= 1.0 and np.min(img_data) >= 0.0:
                img_data_u8 = (img_data * 255).astype(np.uint8)
            else:
                img_data_u8 = np.clip(img_data, 0, 255).astype(np.uint8)

        is_normal_map_preview = False
        if title:
            if "normal" in title.lower() and ("map" in title.lower() or "visual" in title.lower()):
                is_normal_map_preview = True

        pil_image_created = False
        if len(img_data_u8.shape) == 3 and img_data_u8.shape[2] == 3:
            if len(img_data_u8.shape) == 3 and img_data_u8.shape[2] == 3:
                if is_normal_map_preview:
                    nx_channel = img_data_u8[:, :, 0]
                    ny_channel = img_data_u8[:, :, 1]
                    nz_channel = img_data_u8[:, :, 2]
                    rgb_correct_normal_preview = np.stack((nx_channel, ny_channel, nz_channel), axis=-1)
                    display_img_pil = Image.fromarray(rgb_correct_normal_preview, mode='RGB')
                    pil_image_created = True
                else:
                    display_img_pil = Image.fromarray(cv2.cvtColor(img_data_u8, cv2.COLOR_BGR2RGB))
                    pil_image_created = True

        if not pil_image_created:
            if len(img_data_u8.shape) == 3 and img_data_u8.shape[2] == 4:
                display_img_pil = Image.fromarray(cv2.cvtColor(img_data_u8, cv2.COLOR_BGRA2RGBA))
            elif len(img_data_u8.shape) == 2:
                display_img_pil = Image.fromarray(img_data_u8, mode='L')
            else:
                self.clear_canvas(canvas, "Unsupported image format")
                return

        canvas_width = max(1, canvas.winfo_width())
        canvas_height = max(1, canvas.winfo_height())
        img_width, img_height = display_img_pil.size
        if img_width == 0 or img_height == 0: self.clear_canvas(canvas, "Invalid image dimensions"); return

        scale = min(canvas_width / img_width, canvas_height / img_height) if img_width > 0 and img_height > 0 else 1.0
        new_width = max(1, int(img_width * scale))
        new_height = max(1, int(img_height * scale))

        resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
        if not hasattr(Image, 'Resampling'): resample_filter = Image.ANTIALIAS
        try:
            pil_img_resized = display_img_pil.resize((new_width, new_height), resample_filter)
        except Exception:
            pil_img_resized = display_img_pil.resize((new_width, new_height), Image.NEAREST)

        canvas_id = str(id(canvas))
        self.photo_references[canvas_id] = ImageTk.PhotoImage(pil_img_resized)
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.photo_references[canvas_id],
                            anchor=tk.CENTER, tags="img_display_tag")
        if title:
            canvas.create_text(10, 10, text=title, anchor=tk.NW, fill="white", font=("Arial", 9, "bold"),
                               tags="title_text_tag")

    def display_mask(self, wrinkle_mask_data, gradient_mag_data):
        self.mask_fig.clear()
        if wrinkle_mask_data is None or gradient_mag_data is None:
            self.clear_mpl_canvas(self.mask_fig, self.mask_canvas_mpl, "Mask or gradient data unavailable")
            return

        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 2, width_ratios=[1, 1])

        ax1 = self.mask_fig.add_subplot(gs[0])
        ax2 = self.mask_fig.add_subplot(gs[1])

        try:
            im1 = ax1.imshow(gradient_mag_data, cmap='viridis', aspect='equal')
            ax1.set_title('Gradient Magnitude', fontsize=9)
            self.mask_fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        except Exception:
            ax1.text(0.5, 0.5, "Gradient Plot Error", ha='center', va='center', color='red')

        try:
            im2 = ax2.imshow(wrinkle_mask_data, cmap='gray', aspect='equal', vmin=0, vmax=1)
            ax2.set_title('Combined Mask', fontsize=9)
            self.mask_fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        except Exception:
            ax2.text(0.5, 0.5, "Mask Plot Error", ha='center', va='center', color='red')

        for ax_item in [ax1, ax2]:
            ax_item.set_xticks([])
            ax_item.set_yticks([])
            ax_item.set_frame_on(False)

        self.mask_fig.tight_layout(pad=0.5, h_pad=1.0, w_pad=1.0)
        self.mask_canvas_mpl.draw()

    def preview_textures(self):
        if not self.input_file: messagebox.showerror("Error", "Please select an NNRS texture file first"); return
        if self.is_processing: messagebox.showinfo("Info", "Currently processing, please wait"); return

        self.processor.wrinkle_threshold = self.wrinkle_threshold.get()
        self.processor.smoothing_strength = self.smoothing_strength.get()
        self.processor.enable_auto_mask_blur = self.enable_auto_mask_blur.get()
        self.processor.auto_mask_blur_strength = self.auto_mask_blur_strength.get()
        self.processor.roughness_brightness = self.roughness_brightness.get()
        self.processor.roughness_contrast = self.roughness_contrast.get()
        self.processor.specular_brightness = self.specular_brightness.get()
        self.processor.specular_contrast = self.specular_contrast.get()
        self.processor.debug = self.debug_mode.get()
        self.processor.detect_wrinkles_enabled = self.detect_wrinkles_enabled.get()
        self.processor.normal_intensity = self.normal_intensity.get()
        self.processor.normal_mask_mode = self.normal_mask_mode.get()
        self.processor.use_normal_custom_mask = self.use_normal_custom_mask.get()
        if self.use_normal_custom_mask.get() and self.normal_custom_mask is not None:
            self.processor.normal_custom_mask = self.normal_custom_mask
        else:
            self.processor.normal_custom_mask = None
        self.processor.normal_custom_mask_path = self.normal_custom_mask_entry.get() if self.use_normal_custom_mask.get() else None

        self.processor.use_wrinkle_custom_mask = self.use_wrinkle_custom_mask.get()
        self.processor.wrinkle_custom_mask_mode = self.wrinkle_custom_mask_mode.get()
        if self.use_wrinkle_custom_mask.get() and hasattr(self, 'wrinkle_custom_mask') and self.wrinkle_custom_mask is not None:
            self.processor.wrinkle_custom_mask = self.wrinkle_custom_mask.copy()
        else:
            self.processor.wrinkle_custom_mask = None
        self.processor.wrinkle_custom_mask_path = self.wrinkle_custom_mask_entry.get() if self.use_wrinkle_custom_mask.get() else None

        self.processor.use_micronormal = self.use_micronormal.get()
        if self.processor.use_micronormal:
            self.processor.micronormal_strength = self.micronormal_strength.get()
            self.processor.micronormal_tile_size = self.micronormal_tile_size.get()
            self.processor.micronormal_mask_mode = self.micronormal_mask_mode.get()

            if self.use_custom_mask.get() and self.custom_micronormal_mask is not None:
                self.processor.custom_micronormal_mask = self.custom_micronormal_mask
            else:
                self.processor.custom_micronormal_mask = None
            self.processor.custom_micronormal_mask_path = self.custom_mask_entry.get() if self.use_custom_mask.get() else None

            micronormal_path = self.micronormal_entry.get()
            if micronormal_path and (
                    self.processor.micronormal_img is None or self.processor.micronormal_path != micronormal_path):
                try:
                    self.processor.load_micronormal(micronormal_path)
                except Exception as e:
                    messagebox.showerror("Micronormal Error", f"Could not load micronormal map: {e}")
                    return
            elif not micronormal_path:
                 self.processor.micronormal_img = None
                 self.processor.micronormal_path = None

        self.processor.use_roughness_custom_mask = self.use_roughness_custom_mask.get()
        self.processor.roughness_custom_mask_path = self.roughness_custom_mask_path_var.get() if self.use_roughness_custom_mask.get() else None
        self.processor.roughness_custom_mask = self.roughness_custom_mask_gui if self.use_roughness_custom_mask.get() else None
        self.processor.use_specular_custom_mask = self.use_specular_custom_mask.get()
        self.processor.specular_custom_mask_path = self.specular_custom_mask_path_var.get() if self.use_specular_custom_mask.get() else None
        self.processor.specular_custom_mask = self.specular_custom_mask_gui if self.use_specular_custom_mask.get() else None

        if self.hand_drawn_mask_layer1_gui is not None:
            self.processor.set_hand_drawn_mask_layer1(self.hand_drawn_mask_layer1_gui, self.neutral_mask_value)
        else:
            if self.processor.original_img is not None:
                shape = self.processor.original_img.shape[:2]
                neutral_l1 = np.full(shape, self.neutral_mask_value, dtype=np.float32)
                self.processor.set_hand_drawn_mask_layer1(neutral_l1, self.neutral_mask_value)
            else:
                self.processor.hand_drawn_mask_layer1 = None

        self.is_processing = True
        self.update_ui_state(True)
        threading.Thread(target=self._preview_thread, daemon=True).start()

    def _preview_thread(self):
        try:
            self.update_progress("Loading texture...", 5)
            self.update_progress("Splitting channels...", 15)
            self.processor.split_channels()
            self.update_progress("Adjusting Roughness/Specular...", 25)
            self.processor.adjust_roughness_specular()
            self.update_progress("Detecting wrinkles (Layer 0)...", 40)
            self.processor.detect_wrinkles()
            if self.processor.auto_mask_layer0 is not None: self.auto_mask_layer0_gui = self.processor.auto_mask_layer0.copy()
            if self.hand_drawn_mask_layer1_gui is None and self.auto_mask_layer0_gui is not None:
                self.hand_drawn_mask_layer1_gui = np.full_like(self.auto_mask_layer0_gui, self.neutral_mask_value,
                                                               dtype=np.float32)
                self.processor.set_hand_drawn_mask_layer1(self.hand_drawn_mask_layer1_gui, self.neutral_mask_value)
                self.processor._combine_mask_layers()

            if self.processor.use_micronormal and self.processor.micronormal_img is not None:
                self.update_progress("Processing micronormal...", 50)
                self.processor.process_micronormal()

            self.update_progress("Reducing wrinkles (based on combined mask)...", 60)
            self.processor.reduce_wrinkles()

            self.update_progress("Combining channels...", 80)
            self.processor.combine_channels()

            self.mask_has_edits = False
            self.root.after(0, self._update_preview_ui)
            self.update_progress("Preview complete", 100)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Preview Error", f"Preview processing failed: {e}"))
            self.update_progress("Preview failed", 0)
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.update_ui_state(False))

    def update_progress(self, status_text, progress_value):
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, lambda: self.status_var.set(status_text))
            self.root.after(0, lambda: self.progress_var.set(progress_value))

    def update_ui_state(self, processing_state):
        state = tk.DISABLED if processing_state else tk.NORMAL
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, lambda: self.preview_btn.config(state=state))
            self.root.after(0, lambda: self.process_btn.config(state=state))
            normal_btn_state = tk.DISABLED
            if not processing_state and hasattr(self.processor,
                                                'normal_map') and self.processor.normal_map is not None: normal_btn_state = tk.NORMAL
            self.root.after(0, lambda: self.view_normal_btn.config(state=normal_btn_state))

    def view_normal_map(self):
        if hasattr(self.processor, 'processed_normal') and self.processor.processed_normal is not None:
            self.notebook.select(self.processed_normal_frame)
        elif hasattr(self.processor, 'normal_map') and self.processor.normal_map is not None:
            self.notebook.select(self.normal_frame)
        else:
            messagebox.showinfo("No Image", "Normal map data is unavailable.")

    def _update_preview_ui(self):
        self.display_image(self.processor.original_img, self.original_canvas, "Original NNRS Texture (RGBA)")
        self.display_image(self.processor.normal_map, self.normal_canvas, "Original Normal Map (Visual RGB)")
        self.display_image(self.processor.adjusted_roughness_map, self.roughness_canvas_display, "Roughness (Adjusted)")
        self.display_image(self.processor.adjusted_specular_map, self.specular_canvas_display, "Specular (Adjusted)")
        if hasattr(self, 'mask_canvas_mpl'):
            if self.processor.wrinkle_mask is not None and self.processor.gradient_magnitude is not None:
                self.display_mask(self.processor.wrinkle_mask, self.processor.gradient_magnitude)
            else:
                self.clear_mpl_canvas(self.mask_fig, self.mask_canvas_mpl, "Combined mask data unavailable")
        self.display_image(self.processor.processed_img, self.processed_canvas, "Processed NNRS Texture (RGBA)")
        self.display_image(self.processor.processed_normal, self.processed_normal_canvas,
                           "Processed Normal (Visual RGB)")
        if hasattr(self, 'mask_edit_canvas') and self.notebook.select() == str(self.mask_edit_frame):
            self.update_mask_editor_display_with_layers()
        if self.processor.processed_img is not None and hasattr(self, 'processed_frame'):
            self.notebook.select(self.processed_frame)
        elif self.processor.original_img is not None and hasattr(self, 'original_frame'):
            self.notebook.select(self.original_frame)

    def process_textures(self):
        if not self.input_file: messagebox.showerror("Error", "Please select an NNRS texture file first"); return
        self.output_dir = self.output_entry.get()
        self.base_name = self.name_entry.get()
        if not self.output_dir or not self.base_name: messagebox.showerror("Error",
                                                                           "Please specify output directory and file name"); return
        if self.is_processing: messagebox.showinfo("Info", "Currently processing, please wait"); return

        self.processor.wrinkle_threshold = self.wrinkle_threshold.get()
        self.processor.smoothing_strength = self.smoothing_strength.get()
        self.processor.enable_auto_mask_blur = self.enable_auto_mask_blur.get()
        self.processor.auto_mask_blur_strength = self.auto_mask_blur_strength.get()
        self.processor.roughness_brightness = self.roughness_brightness.get()
        self.processor.roughness_contrast = self.roughness_contrast.get()
        self.processor.specular_brightness = self.specular_brightness.get()
        self.processor.specular_contrast = self.specular_contrast.get()
        self.processor.debug = self.debug_mode.get()
        self.processor.detect_wrinkles_enabled = self.detect_wrinkles_enabled.get()
        self.processor.normal_intensity = self.normal_intensity.get()
        self.processor.normal_mask_mode = self.normal_mask_mode.get()
        self.processor.use_normal_custom_mask = self.use_normal_custom_mask.get()
        self.processor.normal_custom_mask_path = self.normal_custom_mask_entry.get() if self.use_normal_custom_mask.get() else None

        if self.use_normal_custom_mask.get() and self.normal_custom_mask is not None:
            self.processor.normal_custom_mask = self.normal_custom_mask
        else:
            self.processor.normal_custom_mask = None

        self.processor.use_wrinkle_custom_mask = self.use_wrinkle_custom_mask.get()
        self.processor.wrinkle_custom_mask_mode = self.wrinkle_custom_mask_mode.get()
        self.processor.wrinkle_custom_mask_path = self.wrinkle_custom_mask_entry.get() if self.use_wrinkle_custom_mask.get() else None

        if self.use_wrinkle_custom_mask.get() and hasattr(self,
                                                          'wrinkle_custom_mask') and self.wrinkle_custom_mask is not None:
            self.processor.wrinkle_custom_mask = self.wrinkle_custom_mask.copy()
        else:
            self.processor.wrinkle_custom_mask = None

        self.processor.use_micronormal = self.use_micronormal.get()
        if self.processor.use_micronormal:
            self.processor.micronormal_strength = self.micronormal_strength.get()
            self.processor.micronormal_tile_size = self.micronormal_tile_size.get()
            self.processor.micronormal_mask_mode = self.micronormal_mask_mode.get()

            new_gui_micronormal_path = self.micronormal_entry.get()

            if new_gui_micronormal_path:
                if self.processor.micronormal_img is None or \
                        (hasattr(self.processor,
                                 'micronormal_path') and self.processor.micronormal_path != new_gui_micronormal_path):
                    try:
                        self.processor.load_micronormal(
                            new_gui_micronormal_path)
                    except Exception as e:
                        self.root.after(0, lambda: messagebox.showwarning("Micronormal Load Error",
                                                                          f"Final processing: Could not load micronormal from {new_gui_micronormal_path}.\nError: {e}. Processing may use previous/no micronormal data."))

            else:
                self.processor.micronormal_img = None
                self.processor.micronormal_path = None

            self.processor.micronormal_path = new_gui_micronormal_path

            self.processor.custom_micronormal_mask_path = self.custom_mask_entry.get() if self.use_micronormal.get() and self.use_custom_mask.get() else None

            if self.use_micronormal.get() and self.use_custom_mask.get() and self.custom_micronormal_mask is not None:
                self.processor.custom_micronormal_mask = self.custom_micronormal_mask
            else:
                self.processor.custom_micronormal_mask = None

        self.processor.use_roughness_custom_mask = self.use_roughness_custom_mask.get()
        self.processor.roughness_custom_mask_path = self.roughness_custom_mask_path_var.get() if self.use_roughness_custom_mask.get() else None
        if self.use_roughness_custom_mask.get() and self.roughness_custom_mask_gui is not None:
            self.processor.roughness_custom_mask = self.roughness_custom_mask_gui
        else:
            self.processor.roughness_custom_mask = None

        self.processor.use_specular_custom_mask = self.use_specular_custom_mask.get()
        self.processor.specular_custom_mask_path = self.specular_custom_mask_path_var.get() if self.use_specular_custom_mask.get() else None
        if self.use_specular_custom_mask.get() and self.specular_custom_mask_gui is not None:
            self.processor.specular_custom_mask = self.specular_custom_mask_gui
        else:
            self.processor.specular_custom_mask = None

        if self.hand_drawn_mask_layer1_gui is not None:
            self.processor.set_hand_drawn_mask_layer1(self.hand_drawn_mask_layer1_gui, self.neutral_mask_value)
        else:
            if self.processor.original_img is not None:
                shape = self.processor.original_img.shape[:2]
                neutral_l1 = np.full(shape, self.neutral_mask_value, dtype=np.float32)
                self.processor.set_hand_drawn_mask_layer1(neutral_l1, self.neutral_mask_value)

        self.is_processing = True
        self.update_ui_state(True)
        threading.Thread(target=self._process_thread, daemon=True).start()

    def _process_thread(self):
        try:
            self.update_progress("Processing texture...", 10)
            _, output_path = self.processor.process(
                self.input_file, output_dir=self.output_dir, base_name=self.base_name,
                save_separate=self.save_separate.get()
            )
            self.root.after(0, self._update_preview_ui)
            if output_path:
                self.update_progress("Processing complete", 100)
                self.root.after(0, lambda: messagebox.showinfo("Success",
                                                               f"Texture processed successfully, saved to:\n{output_path}"))
            else:
                self.update_progress("Processing failed or not saved", 0)
                self.root.after(0,
                                lambda: messagebox.showerror("Error", "Processing or save failed. Check console log."))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Processing Error", f"Processing failed: {e}"))
            self.update_progress("Processing failed", 0)
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.update_ui_state(False))

    def on_closing(self):
        if self.is_processing or (
                hasattr(self.batch_processor, 'is_processing') and self.batch_processor.is_processing):
            if messagebox.askyesno("Confirm Close",
                                   "Processing is currently in progress. Are you sure you want to close?"):
                if hasattr(self.batch_processor, 'cancel'): self.batch_processor.cancel()
                self.root.destroy()
        else:
            self.root.destroy()

    def create_mask_editor_controls(self, parent_controls_frame):
        controls_frame = ttk.Frame(parent_controls_frame)
        controls_frame.pack(pady=2, fill=tk.X)

        left_controls = ttk.Frame(controls_frame)
        left_controls.pack(side=tk.LEFT, padx=(0, 10))
        brush_size_frame = ttk.Frame(left_controls)
        brush_size_frame.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(brush_size_frame, text="Size:").pack(side=tk.LEFT)
        size_values = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150]
        self.size_combobox = ttk.Combobox(brush_size_frame, textvariable=self.brush_size, values=size_values, width=4,
                                          state='readonly')
        self.size_combobox.pack(side=tk.LEFT, padx=(2, 0))
        self.size_combobox.set(self.brush_size.get())

        hardness_frame = ttk.Frame(left_controls)
        hardness_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(hardness_frame, text="Hardness:").pack(side=tk.LEFT)
        hardness_scale = ttk.Scale(hardness_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                   variable=self.brush_hardness, length=60)
        hardness_scale.pack(side=tk.LEFT, padx=(2, 0))
        self.hardness_label_val = ttk.Label(hardness_frame, text=f"{self.brush_hardness.get():.1f}", width=3)
        self.hardness_label_val.pack(side=tk.LEFT)
        hardness_scale.configure(command=lambda v: self.hardness_label_val.config(text=f"{float(v):.1f}"))

        mode_frame = ttk.Frame(left_controls)
        mode_frame.pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="Add", variable=self.brush_mode, value="add").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Remove", variable=self.brush_mode, value="remove").pack(side=tk.LEFT,
                                                                                                  padx=(5, 0))
        ttk.Radiobutton(mode_frame, text="Blur", variable=self.brush_mode, value="blur").pack(side=tk.LEFT, padx=(5, 0))

        right_controls = ttk.Frame(controls_frame)
        right_controls.pack(side=tk.RIGHT)
        action_buttons_frame = ttk.Frame(right_controls)
        action_buttons_frame.pack(padx=(10, 0))
        ttk.Button(action_buttons_frame, text="Reset Layer", command=self.reset_mask_edits, width=10,
                   style='Danger.TButton').pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(action_buttons_frame, text="Apply Layer", command=self.apply_mask_edits, width=10,
                   style='Primary.TButton').pack(side=tk.LEFT)
        self.enhance_mask_editor()

    def init_mask_editor(self):
        if self.mask_editor is None:
            self.mask_editor = MaskEditor(downscale_factor=2.0)

            def on_mask_layer1_changed_in_editor():
                if hasattr(self, 'mask_edit_canvas'):
                    self.update_mask_editor_display_with_layers()
                    self.mask_has_edits = True

            self.mask_editor.on_mask_changed = on_mask_layer1_changed_in_editor

    def on_mask_edit_start(self, event):
        if not hasattr(self, 'mask_editor') or self.mask_editor is None: self.init_mask_editor()
        if self.mask_editor.edit_mask_layer1 is None:
            if self.hand_drawn_mask_layer1_gui is not None:
                self.mask_editor.init_from_mask(self.hand_drawn_mask_layer1_gui.copy(), self.neutral_mask_value)
            else:
                messagebox.showwarning("No Mask", "Layer 1 is not initialized. Please preview first.")
                return

        self.is_drawing = True
        self.last_x, self.last_y = event.x, event.y
        self.mask_editor.add_stroke(event.x, event.y, event.x, event.y, self.brush_size.get(), self.brush_mode.get(),
                                    self.brush_hardness.get())
        self.update_brush_preview(event.x, event.y)

    def on_mask_edit_drag(self, event):
        if not self.is_drawing or not self.mask_editor: return
        self.mask_editor.add_stroke(self.last_x, self.last_y, event.x, event.y, self.brush_size.get(),
                                    self.brush_mode.get(), self.brush_hardness.get())
        self.last_x, self.last_y = event.x, event.y
        self.update_brush_preview(event.x, event.y)

    def on_mask_edit_stop(self, event):
        if not self.is_drawing or not self.mask_editor: return
        self.mask_editor.process_strokes()
        self.is_drawing = False
        self.last_x, self.last_y = None, None
        self.mask_has_edits = True
        self.update_mask_editor_display_with_layers()

    def update_brush_preview(self, x, y):
        self.mask_edit_canvas.delete("brush_preview")
        if self.notebook.select() != str(self.mask_edit_frame): return

        brush_diameter_canvas = self.brush_size.get()
        radius = brush_diameter_canvas / 2
        hardness = self.brush_hardness.get()
        mode = self.brush_mode.get()
        outline_color = "#cccccc"
        if mode == "add":
            outline_color = "#66cc66"
        elif mode == "remove":
            outline_color = "#cc6666"
        elif mode == "blur":
            outline_color = "#6699cc"
        self.mask_edit_canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=outline_color,
                                          width=1, tags="brush_preview", dash=(2, 2))
        if hardness > 0.7:
            self.mask_edit_canvas.create_oval(x - radius + 1, y - radius + 1, x + radius - 1, y + radius - 1,
                                              outline=outline_color, width=1, tags="brush_preview")
        if mode == "blur":
            blur_icon_size = min(10, max(6, int(radius * 0.3)))
            self.mask_edit_canvas.create_text(x, y, text="B", fill=outline_color,
                                              font=("Arial", blur_icon_size, "bold"), tags="brush_preview")

    def remove_brush_preview(self):
        self.mask_edit_canvas.delete("brush_preview")

    def update_mask_editor_display_with_layers(self):
        if not hasattr(self, 'mask_edit_canvas'):
            return

        if not hasattr(self, 'mask_editor') or self.mask_editor is None:
            self.init_mask_editor()

        if self.mask_editor is None:
            self.clear_canvas(self.mask_edit_canvas, "Mask editor instance is missing.")
            return

        if self.hand_drawn_mask_layer1_gui is not None:
            if self.mask_editor.edit_mask_layer1 is None or self.mask_editor.orig_mask_for_editing_layer1 is None:
                self.mask_editor.init_from_mask(self.hand_drawn_mask_layer1_gui.copy(), self.neutral_mask_value)
        elif self.mask_editor.edit_mask_layer1 is None:
            self.clear_canvas(self.mask_edit_canvas, "Layer 1 mask data not available for editor.")
            return

        canvas_width = self.mask_edit_canvas.winfo_width()
        canvas_height = self.mask_edit_canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            return

        display_img = self.mask_editor.get_display_image(
            self.processor.original_img,
            canvas_width,
            canvas_height,
            auto_mask_layer0_for_context=self.auto_mask_layer0_gui
        )

        if display_img is None:
            self.clear_canvas(self.mask_edit_canvas, "Editor data error or original image missing.")
            return

        self.mask_edit_canvas.delete("img_display_tag")
        if len(display_img.shape) == 3 and display_img.shape[2] == 3:
            try:
                display_pil = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
            except cv2.error:
                display_pil = Image.fromarray(display_img)
        elif len(display_img.shape) == 2:
            display_pil = Image.fromarray(display_img, mode='L')
        else:
            display_pil = Image.fromarray(display_img)

        canvas_id = str(id(self.mask_edit_canvas)) + "_edit"
        self.photo_references[canvas_id] = ImageTk.PhotoImage(display_pil)
        self.mask_edit_canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.photo_references[canvas_id], anchor=tk.CENTER,
            tags="img_display_tag"
        )
        self.mask_edit_canvas.create_text(
            10, 10, text="Editing Layer 1 (JET colormap: Red=Add, Blue=Remove/Neutral)",
            anchor=tk.NW, fill="white", font=("Arial", 9, "bold")
        )

    def reset_mask_edits(self):
        if self.auto_mask_layer0_gui is not None:
            if messagebox.askyesno("Reset Manual Edits",
                                   "Are you sure you want to reset manually edited Layer 1 to neutral?"):
                self.hand_drawn_mask_layer1_gui = np.full_like(self.auto_mask_layer0_gui, self.neutral_mask_value,
                                                               dtype=np.float32)
                if self.mask_editor:
                    self.mask_editor.init_from_mask(self.hand_drawn_mask_layer1_gui.copy(), self.neutral_mask_value)
                    self.mask_editor.has_changes = False
                self.mask_has_edits = False
                self.update_mask_editor_display_with_layers()
                self.status_var.set("Manually edited Layer 1 has been reset.")
        else:
            messagebox.showwarning("No Base", "Auto-detected mask (Layer 0) is unavailable. Cannot reset Layer 1.")

    def toggle_micronormal_controls(self):
        state = tk.NORMAL if self.use_micronormal.get() else tk.DISABLED
        self.browse_micronormal_btn.config(state=state)
        self.micronormal_entry.config(state=state)
        self.micro_strength_scale.config(state=state)
        self.micro_tile_scale.config(state=state)
        mask_state = tk.NORMAL if self.use_micronormal.get() else tk.DISABLED
        self.none_mask_radio.config(state=mask_state)
        self.inverse_mask_radio.config(state=mask_state)
        self.direct_mask_radio.config(state=mask_state)
        self.use_custom_mask_check.config(state=mask_state)
        if self.use_micronormal.get() and self.use_custom_mask.get():
            self.custom_mask_entry.config(state=tk.NORMAL)
            self.browse_custom_mask_btn.config(state=tk.NORMAL)
        else:
            self.custom_mask_entry.config(state=tk.DISABLED)
            self.browse_custom_mask_btn.config(state=tk.DISABLED)

    def toggle_normal_custom_mask(self):
        state = tk.NORMAL if self.use_normal_custom_mask.get() else tk.DISABLED
        self.normal_custom_mask_entry.config(state=state)
        self.browse_normal_custom_mask_btn.config(state=state)

    def browse_normal_custom_mask(self):
        file_path = filedialog.askopenfilename(title="Select Custom Mask Texture",
                                               filetypes=[("Image files", "*.png *.jpg *.jpeg *.tga"),
                                                          ("All files", "*.*")])
        if file_path:
            self.normal_custom_mask_entry.delete(0, tk.END)
            self.normal_custom_mask_entry.insert(0, file_path)
            self.normal_custom_mask_path = file_path
            try:
                mask_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is None: raise ValueError("Could not load mask image")
                self.normal_custom_mask = mask_img.astype(np.float32) / 255.0
                self.status_var.set(f"Loaded normal adjustment custom mask: {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Loading Error", f"Could not load custom mask: {e}")

    def browse_wrinkle_custom_mask(self):
        file_path = filedialog.askopenfilename(title="Select Custom Wrinkle Mask File",
                                               filetypes=[("Image files", "*.png *.jpg *.jpeg *.tga"),
                                                          ("All files", "*.*")])
        if file_path:
            self.wrinkle_custom_mask_entry.delete(0, tk.END)
            self.wrinkle_custom_mask_entry.insert(0, file_path)
            try:
                mask_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is None: raise ValueError("Could not load mask image")
                self.wrinkle_custom_mask = mask_img.astype(np.float32) / 255.0
                self.status_var.set(f"Loaded custom wrinkle mask: {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Loading Error", f"Could not load custom mask: {e}")

    def toggle_wrinkle_detection_and_children(self):
        self.toggle_wrinkle_detection()
        self.toggle_auto_mask_blur_controls()

    def toggle_wrinkle_detection(self):
        state = tk.NORMAL if self.detect_wrinkles_enabled.get() else tk.DISABLED
        if hasattr(self, 'threshold_label'):
            self.threshold_label.config(state=state)
            self.threshold_label_val.config(state=state)
            self.threshold_scale.config(state=state)
            self.strength_label.config(state=state)
            self.strength_label_val.config(state=state)
            self.strength_scale.config(state=state)

    def toggle_auto_mask_blur_controls(self):
        parent_enabled = self.detect_wrinkles_enabled.get()
        blur_itself_enabled = self.enable_auto_mask_blur.get()
        final_state = tk.NORMAL if parent_enabled and blur_itself_enabled else tk.DISABLED

        if hasattr(self, 'auto_mask_blur_strength_label'):
            self.auto_mask_blur_strength_label.config(state=final_state)
            self.auto_mask_blur_strength_val_label.config(state=final_state)
            self.auto_mask_blur_strength_scale.config(state=final_state)

        check_state = tk.NORMAL if parent_enabled else tk.DISABLED
        if hasattr(self, 'auto_mask_blur_check'):
            self.auto_mask_blur_check.config(state=check_state)

    def toggle_wrinkle_custom_mask(self):
        state = tk.NORMAL if self.use_wrinkle_custom_mask.get() else tk.DISABLED
        self.wrinkle_custom_mask_entry.config(state=state)
        self.browse_wrinkle_custom_mask_btn.config(state=state)
        self.wrinkle_blend_radio.config(state=state)
        self.wrinkle_replace_radio.config(state=state)
        self.wrinkle_multiply_radio.config(state=state)
        self.wrinkle_subtract_radio.config(state=state)

    def toggle_custom_mask(self):
        state = tk.NORMAL if self.use_custom_mask.get() and self.use_micronormal.get() else tk.DISABLED
        self.custom_mask_entry.config(state=state)
        self.browse_custom_mask_btn.config(state=state)

    def browse_micronormal(self):
        file_path = filedialog.askopenfilename(title="Select Micronormal Texture File",
                                               filetypes=[("Image files", "*.png *.jpg *.jpeg *.tga"),
                                                          ("All files", "*.*")])
        if file_path:
            self.micronormal_entry.delete(0, tk.END)
            self.micronormal_entry.insert(0, file_path)
            self.status_var.set(f"Selected micronormal map: {Path(file_path).name}")

    def browse_custom_mask(self):
        file_path = filedialog.askopenfilename(title="Select Custom Mask Texture",
                                               filetypes=[("Image files", "*.png *.jpg *.jpeg *.tga"),
                                                          ("All files", "*.*")])
        if file_path:
            self.custom_mask_entry.delete(0, tk.END)
            self.custom_mask_entry.insert(0, file_path)
            self.custom_mask_path = file_path
            try:
                mask_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is None: raise ValueError("Could not load mask image")
                self.custom_micronormal_mask = mask_img.astype(np.float32) / 255.0
                self.status_var.set(f"Loaded custom mask: {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Loading Error", f"Could not load custom mask: {e}")

    def apply_mask_edits(self):
        if not self.mask_editor or not self.mask_has_edits:
            messagebox.showinfo("No Changes", "No pending changes in Layer 1 to apply.")
            return
        if messagebox.askyesno("Apply Edits", "Apply current drawing to Layer 1 and re-preview?"):
            final_edited_layer1 = self.mask_editor.get_final_mask()
            if final_edited_layer1 is not None:
                self.hand_drawn_mask_layer1_gui = final_edited_layer1.copy()
                self.mask_has_edits = False
                self.status_var.set("Applied Layer 1 edits. Re-previewing...")
                self.preview_textures()
                self.notebook.select(self.processed_frame)
            else:
                messagebox.showerror("Error", "Could not get final mask from editor.")

    def select_batch_files(self):
        files = filedialog.askopenfilenames(title="Select NNRS Texture Files (Multi-select)",
                                            filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if files:
            for file in files:
                if file not in self.batch_files: self.batch_files.append(file)
            self.update_batch_file_list()

    def select_batch_folder(self):
        folder = filedialog.askdirectory(title="Select Folder Containing NNRS Textures")
        if folder:
            try:
                png_files = sorted(list(Path(folder).glob("*.png")))
                for file_path_obj in png_files:
                    file_str = str(file_path_obj)
                    if file_str not in self.batch_files: self.batch_files.append(file_str)
                self.update_batch_file_list()
                if not png_files: messagebox.showinfo("No Files", f"No PNG files found in {folder}.")
            except Exception as e:
                messagebox.showerror("Folder Error", f"Could not read files from folder: {e}")

    def clear_batch_files(self):
        if self.batch_files and messagebox.askyesno("Clear List", "Are you sure you want to clear the file list?"):
            self.batch_files = []
            self.update_batch_file_list()

    def update_batch_file_list(self):
        self.batch_files_listbox.delete(0, tk.END)
        for file in self.batch_files: self.batch_files_listbox.insert(tk.END, file)
        self.batch_file_count_label.config(text=f"{len(self.batch_files)} files selected")

    def browse_batch_output_dir(self):
        output_dir_selected = filedialog.askdirectory(title="Select Batch Output Directory")
        if output_dir_selected:
            self.batch_output_dir = output_dir_selected
            self.batch_output_entry.delete(0, tk.END)
            self.batch_output_entry.insert(0, self.batch_output_dir)

    def update_batch_threshold_label(self, value):
        self.batch_threshold_label_val.config(text=f"{value:.2f}")

    def update_batch_strength_label(self, value):
        self.batch_strength_label_val.config(text=f"{value:.2f}")

    def toggle_batch_wrinkle_detection_and_children(self):
        self.toggle_batch_wrinkle_detection()
        self.toggle_batch_auto_mask_blur_controls()

    def toggle_batch_wrinkle_detection(self):
        is_enabled_overall = self.batch_detect_wrinkles_enabled.get() and not self.inherit_single_params.get()
        current_state = tk.NORMAL if is_enabled_overall else tk.DISABLED

        if hasattr(self, 'batch_threshold_label'):
            self.batch_threshold_label.config(state=current_state)
            self.batch_threshold_label_val.config(state=current_state)
            self.batch_threshold_scale.config(state=current_state)
            self.batch_strength_label.config(state=current_state)
            self.batch_strength_label_val.config(state=current_state)
            self.batch_strength_scale.config(state=current_state)

    def toggle_batch_auto_mask_blur_controls(self):
        parent_enabled = self.batch_detect_wrinkles_enabled.get()
        blur_itself_enabled = self.batch_enable_auto_mask_blur.get()
        not_inheriting = not self.inherit_single_params.get()

        final_state_children = tk.NORMAL if parent_enabled and blur_itself_enabled and not_inheriting else tk.DISABLED
        final_state_check = tk.NORMAL if parent_enabled and not_inheriting else tk.DISABLED

        if hasattr(self, 'batch_auto_mask_blur_strength_label'):
            self.batch_auto_mask_blur_strength_label.config(state=final_state_children)
            self.batch_auto_mask_blur_strength_val_label.config(state=final_state_children)
            self.batch_auto_mask_blur_strength_scale.config(state=final_state_children)

        if hasattr(self, 'batch_auto_mask_blur_check'):
            self.batch_auto_mask_blur_check.config(state=final_state_check)

    def toggle_batch_wrinkle_custom_mask(self):
        is_enabled_overall = self.batch_use_wrinkle_custom_mask.get() and not self.inherit_single_params.get()
        state = tk.NORMAL if is_enabled_overall else tk.DISABLED

        self.batch_wrinkle_custom_mask_entry.config(state=state)
        self.batch_browse_wrinkle_custom_mask_btn.config(state=state)
        self.batch_wrinkle_blend_radio.config(state=state)
        self.batch_wrinkle_replace_radio.config(state=state)
        self.batch_wrinkle_multiply_radio.config(state=state)
        self.batch_wrinkle_subtract_radio.config(state=state)

    def toggle_batch_micronormal_controls(self):
        is_enabled_overall = self.batch_use_micronormal.get() and not self.inherit_single_params.get()
        state = tk.NORMAL if is_enabled_overall else tk.DISABLED

        self.batch_micronormal_entry.config(state=state)
        self.batch_browse_micronormal_btn.config(state=state)
        self.batch_micro_strength_scale.config(state=state)
        self.batch_micro_tile_scale.config(state=state)

        if hasattr(self, 'bmicro_mask_options_frame'):
            for widget in self.bmicro_mask_options_frame.winfo_children():
                if isinstance(widget, (ttk.Radiobutton, ttk.Checkbutton)):
                    widget.config(state=state)

        self.toggle_batch_custom_mask()

    def toggle_batch_custom_mask(self):
        is_enabled_overall = self.batch_use_custom_mask.get() and self.batch_use_micronormal.get() and not self.inherit_single_params.get()
        state = tk.NORMAL if is_enabled_overall else tk.DISABLED

        self.batch_custom_mask_entry.config(state=state)
        self.batch_browse_custom_mask_btn.config(state=state)

    def browse_batch_micronormal(self):
        file_path = filedialog.askopenfilename(title="Select Micronormal Texture File",
                                               filetypes=[("Image files", "*.png *.jpg *.jpeg *.tga"),
                                                          ("All files", "*.*")])
        if file_path: self.batch_micronormal_entry.delete(0, tk.END); self.batch_micronormal_entry.insert(0,
                                                                                                          file_path); self.batch_micronormal_path = file_path

    def browse_batch_custom_mask(self):
        file_path = filedialog.askopenfilename(title="Select Custom Mask Texture",
                                               filetypes=[("Image files", "*.png *.jpg *.jpeg *.tga"),
                                                          ("All files", "*.*")])
        if file_path: self.batch_custom_mask_entry.delete(0, tk.END); self.batch_custom_mask_entry.insert(0,
                                                                                                          file_path); self.batch_custom_mask_path = file_path

    def browse_batch_wrinkle_custom_mask(self):
        file_path = filedialog.askopenfilename(title="Select Custom Wrinkle Mask File",
                                               filetypes=[("Image files", "*.png *.jpg *.jpeg *.tga"),
                                                          ("All files", "*.*")])
        if file_path: self.batch_wrinkle_custom_mask_entry.delete(0,
                                                                  tk.END); self.batch_wrinkle_custom_mask_entry.insert(
            0, file_path); self.batch_wrinkle_custom_mask_path = file_path

    def toggle_batch_normal_custom_mask(self):
        is_enabled_overall = self.batch_use_normal_custom_mask.get() and not self.inherit_single_params.get()
        state = tk.NORMAL if is_enabled_overall else tk.DISABLED
        self.batch_normal_custom_mask_entry.config(state=state)
        self.batch_browse_normal_custom_mask_btn.config(state=state)

    def browse_batch_normal_custom_mask(self):
        file_path = filedialog.askopenfilename(title="Select Custom Mask Texture for Normal Adjustment",
                                               filetypes=[("Image files", "*.png *.jpg *.jpeg *.tga"),
                                                          ("All files", "*.*")])
        if file_path: self.batch_normal_custom_mask_entry.delete(0, tk.END); self.batch_normal_custom_mask_entry.insert(
            0, file_path); self.batch_normal_custom_mask_path = file_path

    def start_batch_processing(self):
        if not self.batch_files: messagebox.showerror("Error", "Please select files to process first"); return
        self.batch_output_dir = self.batch_output_entry.get()
        if not self.batch_output_dir or not os.path.isdir(self.batch_output_dir): messagebox.showerror("Error",
                                                                                                       "Please select a valid output directory"); return

        self.batch_start_btn.config(state=tk.DISABLED)
        self.batch_cancel_btn.config(state=tk.NORMAL)
        self.batch_log_text.config(state=tk.NORMAL)
        self.batch_log_text.delete(1.0, tk.END)
        self.batch_log_text.insert(tk.END,
                                   f"Starting batch process for {len(self.batch_files)} files...\nOutput to: {self.batch_output_dir}\n---\n")
        self.batch_log_text.config(state=tk.DISABLED)

        self.batch_processor.set_files(self.batch_files)
        self.batch_processor.set_callback(self.batch_progress_callback)

        if self.use_mask_template_for_batch.get() and self.hand_drawn_mask_layer1_gui is not None:
            self.batch_processor.mask_template_layer1 = self.hand_drawn_mask_layer1_gui.copy()
            self._log_batch_message("Using current manually edited Layer 1 as template for processing.\n")
        else:
            self.batch_processor.mask_template_layer1 = None
            if self.use_mask_template_for_batch.get(): self._log_batch_message(
                "Warning: Mask template selected, but Layer 1 is unavailable. Template will not be used.\n")

        processor_params = {}
        if self.inherit_single_params.get():
            self._log_batch_message("Using parameters inherited from Single File settings.\n")
            processor_params = {
                'wrinkle_threshold': self.wrinkle_threshold.get(),
                'smoothing_strength': self.smoothing_strength.get(),
                'detect_wrinkles_enabled': self.detect_wrinkles_enabled.get(),
                'enable_auto_mask_blur': self.enable_auto_mask_blur.get(),
                'auto_mask_blur_strength': self.auto_mask_blur_strength.get(),
                'use_wrinkle_custom_mask': self.use_wrinkle_custom_mask.get(),
                'wrinkle_custom_mask_path': self.wrinkle_custom_mask_entry.get() if self.use_wrinkle_custom_mask.get() else None,
                'wrinkle_custom_mask_mode': self.wrinkle_custom_mask_mode.get(),
                'roughness_brightness': self.roughness_brightness.get(),
                'roughness_contrast': self.roughness_contrast.get(),
                'use_roughness_custom_mask': self.use_roughness_custom_mask.get(),
                'roughness_custom_mask_path': self.roughness_custom_mask_path_var.get() if self.use_roughness_custom_mask.get() else None,
                'specular_brightness': self.specular_brightness.get(),
                'specular_contrast': self.specular_contrast.get(),
                'use_specular_custom_mask': self.use_specular_custom_mask.get(),
                'specular_custom_mask_path': self.specular_custom_mask_path_var.get() if self.use_specular_custom_mask.get() else None,
                'normal_intensity': self.normal_intensity.get(),
                'normal_mask_mode': self.normal_mask_mode.get(),
                'use_normal_custom_mask': self.use_normal_custom_mask.get(),
                'normal_custom_mask_path': self.normal_custom_mask_entry.get() if self.use_normal_custom_mask.get() else None,
                'use_micronormal': self.use_micronormal.get(),
                'micronormal_path': self.micronormal_entry.get() if self.use_micronormal.get() else None,
                'micronormal_strength': self.micronormal_strength.get(),
                'micronormal_tile_size': self.micronormal_tile_size.get(),
                'micronormal_mask_mode': self.micronormal_mask_mode.get(),
                'custom_micronormal_mask_path': self.custom_mask_entry.get() if self.use_micronormal.get() and self.use_custom_mask.get() else None,

                'debug': self.debug_mode.get()
            }
        else:
            self._log_batch_message("Using parameters from Batch Process settings.\n")
            processor_params = {
                'wrinkle_threshold': self.batch_wrinkle_threshold.get(),
                'smoothing_strength': self.batch_smoothing_strength.get(),
                'detect_wrinkles_enabled': self.batch_detect_wrinkles_enabled.get(),
                'enable_auto_mask_blur': self.batch_enable_auto_mask_blur.get(),
                'auto_mask_blur_strength': self.batch_auto_mask_blur_strength.get(),
                'use_wrinkle_custom_mask': self.batch_use_wrinkle_custom_mask.get(),
                'wrinkle_custom_mask_path': self.batch_wrinkle_custom_mask_entry.get() if self.batch_use_wrinkle_custom_mask.get() else None,
                'wrinkle_custom_mask_mode': self.batch_wrinkle_custom_mask_mode.get(),
                'roughness_brightness': self.batch_roughness_brightness.get(),
                'roughness_contrast': self.batch_roughness_contrast.get(),
                'use_roughness_custom_mask': self.batch_use_roughness_custom_mask.get(),
                'roughness_custom_mask_path': self.batch_roughness_custom_mask_path_var.get() if self.batch_use_roughness_custom_mask.get() else None,
                'specular_brightness': self.batch_specular_brightness.get(),
                'specular_contrast': self.batch_specular_contrast.get(),
                'use_specular_custom_mask': self.batch_use_specular_custom_mask.get(),
                'specular_custom_mask_path': self.batch_specular_custom_mask_path_var.get() if self.batch_use_specular_custom_mask.get() else None,
                'normal_intensity': self.batch_normal_intensity.get(),
                'normal_mask_mode': self.batch_normal_mask_mode.get(),
                'use_normal_custom_mask': self.batch_use_normal_custom_mask.get(),
                'normal_custom_mask_path': self.batch_normal_custom_mask_entry.get() if self.batch_use_normal_custom_mask.get() else None,
                'use_micronormal': self.batch_use_micronormal.get(),
                'micronormal_path': self.batch_micronormal_entry.get() if self.batch_use_micronormal.get() else None,
                'micronormal_strength': self.batch_micronormal_strength.get(),
                'micronormal_tile_size': self.batch_micronormal_tile_size.get(),
                'micronormal_mask_mode': self.batch_micronormal_mask_mode.get(),
                'custom_micronormal_mask_path': self.batch_custom_mask_entry.get() if self.batch_use_micronormal.get() and self.batch_use_custom_mask.get() else None,

                'debug': self.batch_debug_mode.get()
            }

        processor_params['save_separate'] = self.batch_save_separate.get()
        processor_params['neutral_mask_value'] = self.neutral_mask_value

        if processor_params['use_wrinkle_custom_mask'] and not processor_params['wrinkle_custom_mask_path']:
            self._log_batch_message("Warning: 'Use Custom Wrinkle Mask' is checked, but no file path is provided.\n")
            processor_params['use_wrinkle_custom_mask'] = False

        if processor_params['use_roughness_custom_mask'] and not processor_params['roughness_custom_mask_path']:
            self._log_batch_message(
                "Warning: 'Use Custom Mask' for Roughness is checked, but no file path is provided.\n")
            processor_params['use_roughness_custom_mask'] = False

        if processor_params['use_specular_custom_mask'] and not processor_params['specular_custom_mask_path']:
            self._log_batch_message(
                "Warning: 'Use Custom Mask' for Specular is checked, but no file path is provided.\n")
            processor_params['use_specular_custom_mask'] = False

        if processor_params['use_normal_custom_mask'] and not processor_params['normal_custom_mask_path']:
            self._log_batch_message(
                "Warning: 'Use Custom Mask' for Normal Intensity is checked, but no file path is provided.\n")
            processor_params['use_normal_custom_mask'] = False

        if processor_params['use_micronormal']:
            if not processor_params['micronormal_path']:
                self._log_batch_message(
                    "Warning: 'Enable Micronormal Overlay' is checked, but no Micronormal Map path is provided.\n")

            if processor_params.get('custom_micronormal_mask_path') is None and \
                    ((self.inherit_single_params.get() and self.use_custom_mask.get()) or \
                     (
                             not self.inherit_single_params.get() and self.batch_use_custom_mask.get())):
                self._log_batch_message(
                    "Warning: 'Use Custom Mask' for Micronormal is checked, but no file path is provided.\n")


        threading.Thread(target=self.batch_processor.process, args=(processor_params, self.batch_output_dir),
                         daemon=True).start()
        self.update_batch_progress()

    def _log_batch_message(self, message):
        if hasattr(self, 'batch_log_text') and self.batch_log_text.winfo_exists():
            self.batch_log_text.config(state=tk.NORMAL)
            self.batch_log_text.insert(tk.END, message)
            self.batch_log_text.see(tk.END)
            self.batch_log_text.config(state=tk.DISABLED)

    def cancel_batch_processing(self):
        if self.batch_processor.is_processing:
            self.batch_processor.cancel()
            self._log_batch_message("User requested to cancel batch processing...\n")
            self.batch_cancel_btn.config(state=tk.DISABLED)

    def batch_progress_callback(self, event_type, current_index, total_files, file_path, data):
        self.batch_progress_queue.put(
            {'event_type': event_type, 'current_index': current_index, 'total_files': total_files,
             'file_path': file_path, 'data': data})

    def update_batch_progress(self):
        try:
            while not self.batch_progress_queue.empty():
                update = self.batch_progress_queue.get_nowait()
                event_type = update['event_type']
                current_idx = update['current_index']
                total = update['total_files']
                f_path = update['file_path']
                data_val = update['data']
                file_name = os.path.basename(f_path) if f_path else "N/A"
                overall_progress = (current_idx / total) * 100 if total > 0 else 0

                if event_type == 'processing':
                    self.batch_current_file_label.config(text=f"Processing ({current_idx + 1}/{total}): {file_name}")
                    self.batch_progress_var.set(overall_progress)
                    self._log_batch_message(f"[{current_idx + 1}/{total}] Started processing: {file_name}\n")
                elif event_type == 'completed':
                    self.batch_progress_var.set(((current_idx + 1) / total) * 100 if total > 0 else 0)
                    self._log_batch_message(
                        f"✓ Completed: {file_name} -> {os.path.basename(str(data_val)) if data_val else 'N/A'}\n")
                elif event_type == 'error':
                    self._log_batch_message(f"✗ Error: {file_name}\n  Reason: {data_val}\n")
                elif event_type == 'warning':
                    self._log_batch_message(f"⚠ Warning: {file_name}\n  Details: {data_val}\n")
                elif event_type == 'finished':
                    self.batch_progress_var.set(100)
                    self.batch_current_file_label.config(text="Batch processing finished!")
                    results = data_val
                    success_count = sum(1 for r in results if r['status'] == 'success')
                    error_count = sum(1 for r in results if r['status'] == 'error')
                    cancelled_count = sum(1 for r in results if r['status'] == 'cancelled')
                    summary = f"\n--- Batch Process Finished ---\nSuccessful: {success_count}, Failed: {error_count}, Cancelled: {cancelled_count}\n"
                    self._log_batch_message(summary)
                    messagebox.showinfo("Batch Complete", summary.strip())
                    self.batch_start_btn.config(state=tk.NORMAL)
                    self.batch_cancel_btn.config(state=tk.DISABLED)
                    return
        except queue.Empty:
            pass
        except Exception:
            pass

        if self.batch_processor.is_processing or not self.batch_progress_queue.empty():
            self.root.after(200, self.update_batch_progress)
        elif not self.batch_processor.is_processing and self.batch_start_btn['state'] == tk.DISABLED:
            self.batch_start_btn.config(state=tk.NORMAL)
            self.batch_cancel_btn.config(state=tk.DISABLED)
            if self.batch_current_file_label.cget("text") != "Batch processing finished!":
                self.batch_current_file_label.config(text="Batch process ended (status may vary)")

    def show_tooltip(self, widget, text):
        tooltip = tk.Toplevel(self.root)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry("+0+0")
        label = ttk.Label(tooltip, text=text, justify=tk.LEFT, background="#FFFFDD", relief=tk.SOLID, borderwidth=1,
                          font=("Arial", "8", "normal"))
        label.pack(ipadx=5, ipady=2)
        tooltip.withdraw()

        def enter(event): x, y = self.root.winfo_pointerxy(); tooltip.wm_geometry(
            f"+{x + 15}+{y + 10}"); tooltip.deiconify()

        def leave(event): tooltip.withdraw()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
        return tooltip

    def improve_ui_feedback(self):
        pass

    def enhance_mask_editor(self):
        guide_frame = ttk.Frame(self.mask_edit_controls)
        guide_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))
        ttk.Label(guide_frame, text="Hint: Left-click to draw, mouse wheel to change brush size (within edit area)",
                  foreground="#666666", font=("Arial", 8, "italic")).pack(side=tk.LEFT)

    def apply_brush_preset(self, size, hardness):
        self.brush_size.set(size)
        self.brush_hardness.set(hardness)
        self.size_combobox.set(size)
        self.hardness_label_val.config(text=f"{hardness:.1f}")

    def set_blur_strength(self, hardness):
        self.brush_hardness.set(hardness)
        self.hardness_label_val.config(text=f"{hardness:.1f}")
        self.brush_mode.set("blur")

    def create_toolbar(self, parent):
        self.toolbar = ttk.Frame(parent, style="Card.TFrame")
        self.toolbar.pack(side=tk.TOP, fill=tk.X, padx=0, pady=(0, 5))
        ttk.Button(self.toolbar, text="Open", style="Primary.TButton", width=8, command=self.browse_file).pack(
            side=tk.LEFT, padx=2, pady=2)
        ttk.Button(self.toolbar, text="Save", style="Secondary.TButton", width=8, command=self.process_textures).pack(
            side=tk.LEFT, padx=2, pady=2)
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=3)
        ttk.Button(self.toolbar, text="Help", width=8, command=lambda: messagebox.showinfo("Help",
                                                                                           "NNRS Texture Wrinkle Processing Tool v1.1 (Layers)\n\nLayer System:\n- Layer 0: Automatically detected wrinkles.\n- Layer 1: Manually edited wrinkles (via 'Edit Manual Mask' tab).\nFinal effect is a combination of both layers.")).pack(
            side=tk.RIGHT, padx=2, pady=2)
        return self.toolbar


if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ttkthemes import ThemedTk

        s = ttk.Style()
        if 'clam' in s.theme_names():
            s.theme_use('clam')
        elif 'alt' in s.theme_names():
            s.theme_use('alt')
        elif 'default' in s.theme_names():
            s.theme_use('default')

    except ImportError:
        pass
    app = NNRSTextureGUI(root)
    root.mainloop()