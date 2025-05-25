# Ob_NNRS Skin Texture Wrinkle Processing Tool

## Description

This Python application provides a graphical user interface (GUI) for processing skin textures, specifically targeting the detection, editing, and reduction of wrinkles. It's designed to work with textures that likely pack normal map data (X and Y components) along with roughness and specular information into RGBA channels (e.g., Normal X in Red, Normal Y in Green, Roughness in Blue, Specular in Alpha).

The tool offers both automated processing and manual control, allowing users to refine wrinkle masks and adjust various texture parameters.

## Features

* **Advanced Wrinkle Detection**: Automatically identifies wrinkle patterns in textures using image analysis techniques based on normal map data.
* **Interactive Mask Editing**:
  * A dedicated mask editing tab for fine-tuning wrinkle masks.
  * Brush tools: Add, Remove, and Blur wrinkle effects.
  * Adjustable brush size and hardness.
  * Real-time preview of edits on the texture.
* **Layered Mask System**:
  * **Layer 0**: Automatically detected wrinkle mask.
  * **Layer 1**: Manually painted adjustments, blended with Layer 0.
* **Wrinkle Reduction**: Sophisticated smoothing algorithms (Bilateral filter, Edge-Preserving filter, Gaussian blur) applied to normal map data, controlled by the combined wrinkle mask and smoothing strength.
* **Micronormal Overlay**: Ability to overlay a tiling micronormal map onto the base normal map, with controls for strength, tile size, and masking.
* **Texture Adjustments**:
  * Modify normal map intensity with masking options.
  * Adjust brightness and contrast for roughness and specular maps.
  * Option to use custom grayscale masks for targeted adjustments.
* **Single File Processing**: Load, process, and save individual texture files.
* **Batch Processing**:
  * Process multiple files from a selection or an entire folder.
  * Option to inherit processing parameters from the single-file settings.
  * Use the currently edited manual mask (Layer 1) as a template for batch operations.
  * Progress bar and logging for batch operations.
* **Visual Feedback**: Multiple tabs in the GUI to display:
  * Original loaded texture.
  * Extracted normal map, roughness, and specular maps.
  * Combined wrinkle mask (automatic + manual) and gradient magnitude.
  * Loaded micronormal map.
  * Final processed texture and processed normal map.
* **Save Options**:
  * Save the final processed texture (RGBA).
  * Optionally save separate texture components (processed normal, adjusted roughness, adjusted specular, combined mask).

## Key Components/Classes

* **`NNRSTextureGUI`**: The main class that builds and manages the Tkinter graphical user interface, handles user interactions, and orchestrates the processing.
* **`NNRSTextureProcessor`**: Core class responsible for all texture analysis and processing logic. This includes loading textures, splitting channels, detecting wrinkles, applying smoothing, handling micronormals, and combining channels back.
* **`MaskEditor`**: Manages the interactive editing of the manual wrinkle mask (Layer 1). It handles brush strokes, downscales masks for performance during editing, and generates display images for the editor.
* **`BatchProcessor`**: Handles the logic for processing multiple files sequentially, managing parameters, and providing progress feedback.
* **`PerformanceTracker`**: A utility class (though its `points_buffer` seems unused in the current version) for managing update rates, potentially for UI elements or drawing operations.

## Usage Overview

### Single File Processing

1. **Load Texture**: Use the "Browse..." button in the "File Selection" section or the "Open" button in the toolbar to load an NNRS texture file (typically a `.png` with RGBA channels).
2. **Adjust Parameters**:
    * Modify wrinkle detection sensitivity, normal smoothing strength, and auto-mask blur settings.
    * Load custom masks for wrinkle detection, roughness/specular adjustments, normal intensity, or micronormal blending if needed.
    * Adjust roughness/specular brightness and contrast.
    * Configure normal intensity and its masking mode.
    * Enable and configure micronormal overlay parameters.
3. **Edit Manual Mask (Layer 1)** (Optional):
    * Navigate to the "Edit Manual Mask (Layer 1)" tab.
    * Use the brush tools (Add, Remove, Blur) with adjustable size and hardness to paint on the mask. Red areas in the JET colormap visualization generally indicate additions, while blue indicates removals or neutral areas.
    * The underlying original texture and the auto-detected mask (Layer 0) are shown for context.
    * Click "Apply Layer" to apply your manual edits. This will update the internal Layer 1 mask and trigger a re-preview.
    * "Reset Layer" will revert Layer 1 to a neutral state.
4. **Preview**: Click the "Preview" button to see the results of the current settings on the various output tabs ("Processed Texture", "Processed Normal", "Combined Mask", etc.) without saving.
5. **Process & Save**:
    * Specify the output directory and file name in the "Save Options" section.
    * Click "Process & Save" (or "Save" in the toolbar) to apply the processing and save the output files.
    * Choose "Save Separate Textures" to output individual maps.

### Batch Processing

1. **Select Files/Folder**: In the "Batch Process" tab, use "Select Files..." or "Select Folder..." to add textures to the batch list.
2. **Set Output Directory**: Specify a directory where processed files will be saved.
3. **Configure Parameters**:
    * Either check "Inherit Single File Parameters" to use the settings from the "Single File" tab.
    * Or, uncheck it and configure the batch-specific parameters below it.
    * Optionally, check "Use Current Edited Mask (Layer 1) as Template" if you have a manually edited mask you wish to apply (resized as needed) to all images in the batch.
4. **Start Batch**: Click "Start Batch Process". Progress and logs will be displayed.
5. **Cancel Batch**: Click "Cancel" to stop an ongoing batch process.

## Parameters Overview

The tool offers a wide range of parameters, broadly categorized into:

* **Wrinkle Detection**: Sensitivity of the automatic detection.
* **Normal Smoothing**: Strength of the smoothing applied to reduce wrinkles in the normal map.
* **Auto-Mask (Layer 0) Blurring**: Controls for blurring the automatically generated wrinkle mask.
* **Custom Masks**: Options to load external grayscale masks to influence:
  * Wrinkle detection (blend, replace, multiply, subtract modes).
  * Roughness/Specular adjustments.
  * Normal intensity adjustments.
  * Micronormal blending.
* **Roughness & Specular**: Brightness and contrast controls.
* **Normal Intensity**: Overall strength of the normal map details, with masking options (none, inverse of wrinkle mask, direct wrinkle mask, custom mask).
* **Micronormal Overlay**: Enabling, map selection, blend strength, tile size, and masking options.
* **Save Options**: Saving separate texture channels, debug mode (saves intermediate masks).

Experiment with these parameters to achieve the desired results for your specific textures.

```

```
