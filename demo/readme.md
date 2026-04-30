## Burn TBSA Calculator - Gradio Application

Interactive burn area segmentation and TBSA estimation tool using SAM2 and BurnAreaNet.

### 1. Install Dependencies

```bash
# Install SAM2
pip install git+https://github.com/facebookresearch/sam2.git

# Install other dependencies
pip install -r requirements.txt
```

### 2. Prepare Model Weights

#### SAM2 Checkpoint

Download the SAM2.1 Hiera Tiny, place `sam2.1_hiera_tiny.pt` in the `demo/` directory.

#### BurnAreaNet Model

The trained BurnAreaNet model (`model.safetensors`) should be located in the project root. We recommend you use ``SAM_P_MHB`` version. Create a symbolic link from the parent directory:
```bash
ln -s ../model.safetensors model.safetensors
```

### 3. Run the Application

```bash
python gradio_app.py
```

The application will start at **http://localhost:7860**.

### 4. Usage

1. **Load Images**: Upload front and back body photos, or click "Load Examples" to use sample images
2. **Body Segmentation** (default stage):
   - Select image side (front/back)
   - Click two points on the image to define a bounding box (top-left and bottom-right)
   - SAM2 will segment the body region (green overlay)
   - Clicking a third time resets and starts over
3. **Burn Area Segmentation**:
   - Switch stage to "burn"
   - Click two points per burn area to define each region
   - Each pair of clicks creates a new burn area group
   - Select a burn area in the list to highlight it (shows red points + box)
   - Use "Delete Selected Burn Area" to remove unwanted regions
4. **Calculate TBSA**: Once both sides have body masks, click "Calculate TBSA" to get the estimated percentage (burn masks on either side are optional)

### Known Limitations

- **Vanilla SAM2.1** does not segment human body and burn areas well enough for production use; it is provided for demonstration purposes only.
- **BurnAreaNet** performs poorly on cases with concentrated burns on one side (e.g., a single side entirely burned), as such scenarios are not covered in the training data. This will be addressed in future work.
- **BurnAreaNet** always predicts a TBSA greater than 0. If no burn masks are provided for either side, the model treats it as burns being invisible (not absent) in the given images, and will still output a non-zero TBSA estimation.
