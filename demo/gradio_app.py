import gradio as gr
import torch
import numpy as np
from PIL import Image
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.transforms import v2 as T
from burn_area_net import BurnAreaNet
from safetensors.torch import load_file
import os

# Detect device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print("Using GPU for inference")
else:
    device = torch.device("cpu")
    print("Warning: No GPU detected. Using CPU for inference.")
    print("Warning: Inference will be slow. For better performance, please use a GPU.")

print(f"Using device: {device}")

# Initialize models
print("Loading SAM2 model...")
sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt")
sam2_model.to(device)
sam2predictor = SAM2ImagePredictor(sam2_model)

print("Loading BurnAreaNet model...")
burn_net = BurnAreaNet()
burn_net.to(device)
if os.path.exists("model.safetensors"):
    state_dict = load_file("model.safetensors")
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    burn_net.load_state_dict(state_dict)
    burn_net.eval()
    print("BurnAreaNet model loaded successfully.")
else:
    print("Warning: model.safetensors not found.")

# Image preprocessing transform
transform = T.Compose(
    [
        T.ToImage(),
        T.ToDtype(torch.uint8, scale=True),
        T.Resize((224, 224)),
        T.ToDtype(torch.float32, scale=True),
    ]
)


# Global state
class AppState:
    def __init__(self):
        self.front_image = None
        self.back_image = None
        self.current_side = "front"
        self.stage = "body"
        self.front_body_points = []
        self.front_body_mask = None
        self.back_body_points = []
        self.back_body_mask = None
        self.front_burn_groups = []
        self.back_burn_groups = []
        self.front_burn_current_points = []
        self.back_burn_current_points = []
        self.selected_burn_idx = -1


state = AppState()


def reset_state():
    """Reset all segmentation state"""
    state.front_body_points = []
    state.front_body_mask = None
    state.back_body_points = []
    state.back_body_mask = None
    state.front_burn_groups = []
    state.back_burn_groups = []
    state.front_burn_current_points = []
    state.back_burn_current_points = []
    state.selected_burn_idx = -1


def get_current_image():
    if state.current_side == "front":
        return state.front_image
    return state.back_image


def get_burn_groups():
    if state.current_side == "front":
        return state.front_burn_groups
    return state.back_burn_groups


def get_burn_current_points():
    if state.current_side == "front":
        return state.front_burn_current_points
    return state.back_burn_current_points


def get_burn_radio_choices():
    groups = get_burn_groups()
    if len(groups) == 0:
        return gr.update(choices=[], value=None)
    choices = [f"Burn Area {i+1}" for i in range(len(groups))]
    current_val = (
        f"Burn Area {state.selected_burn_idx + 1}"
        if 0 <= state.selected_burn_idx < len(groups)
        else None
    )
    return gr.update(choices=choices, value=current_val)


def get_current_mask_preview():
    image = get_current_image()
    if image is None:
        return None
    image = image.copy()

    if state.stage == "body":
        points = (
            state.front_body_points
            if state.current_side == "front"
            else state.back_body_points
        )
        mask = (
            state.front_body_mask
            if state.current_side == "front"
            else state.back_body_mask
        )
        if mask is not None:
            colored_mask = np.zeros_like(image)
            colored_mask[mask[0] > 0] = [0, 255, 0]
            image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        elif len(points) == 1:
            px, py = points[0]
            cv2.circle(image, (px, py), 4, (0, 255, 255), -1)

    elif state.stage == "burn":
        groups = get_burn_groups()
        current_pts = get_burn_current_points()

        if len(groups) > 0:
            colored_mask = np.zeros_like(image)
            for group in groups:
                m = group["mask"]
                if m is not None:
                    colored_mask[m[0] > 0] = [0, 255, 0]
            image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

        if 0 <= state.selected_burn_idx < len(groups):
            sel = groups[state.selected_burn_idx]
            pts = sel["points"]
            for px, py in pts:
                cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
                cv2.circle(image, (px, py), 5, (0, 0, 200), 2)
            cv2.rectangle(
                image, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), (0, 0, 255), 2
            )

        if len(current_pts) == 1:
            px, py = current_pts[0]
            cv2.circle(image, (px, py), 4, (0, 255, 255), -1)

    return image


def load_images(front_path, back_path):
    reset_state()
    if front_path:
        state.front_image = np.array(Image.open(front_path).convert("RGB"))
    if back_path:
        state.back_image = np.array(Image.open(back_path).convert("RGB"))
    # Set SAM2 image for default side (front)
    if state.front_image is not None:
        with torch.inference_mode():
            sam2predictor.set_image(state.front_image)
    return get_current_mask_preview(), get_burn_radio_choices()


def load_example_images():
    reset_state()
    front_path = "images/front.png"
    back_path = "images/back.png"
    state.front_image = np.array(Image.open(front_path).convert("RGB"))
    state.back_image = np.array(Image.open(back_path).convert("RGB"))
    # Set SAM2 image for default side (front)
    with torch.inference_mode():
        sam2predictor.set_image(state.front_image)
    return front_path, back_path, get_current_mask_preview(), get_burn_radio_choices()


def set_current_side(side):
    state.current_side = side
    state.selected_burn_idx = -1
    # Set SAM2 image for the selected side
    image = get_current_image()
    if image is not None:
        with torch.inference_mode():
            sam2predictor.set_image(image)
    return get_current_mask_preview(), get_burn_radio_choices()


def set_stage(stage):
    state.stage = stage
    return (
        get_current_mask_preview(),
        get_burn_radio_choices(),
        gr.update(visible=(stage == "burn")),
    )


def select_burn_region(evt: gr.SelectData):
    if evt.index is not None:
        state.selected_burn_idx = evt.index
    return get_current_mask_preview()


def delete_burn_region(selected):
    groups = get_burn_groups()
    if selected is None or len(groups) == 0:
        return get_current_mask_preview(), get_burn_radio_choices()
    try:
        idx = int(selected.split(" ")[-1]) - 1
    except (ValueError, IndexError):
        return get_current_mask_preview(), get_burn_radio_choices()
    if 0 <= idx < len(groups):
        groups.pop(idx)
        state.selected_burn_idx = -1
    return get_current_mask_preview(), get_burn_radio_choices()


def predict_sam2(box):
    """Run SAM2 prediction with the pre-set image."""
    with torch.inference_mode():
        masks, _, _ = sam2predictor.predict(box=box, multimask_output=False)
    return masks


def add_point(evt: gr.SelectData):
    if sam2predictor is None:
        return get_current_mask_preview(), get_burn_radio_choices()
    image = get_current_image()
    if image is None:
        return None, get_burn_radio_choices()

    x, y = evt.index[0], evt.index[1]

    if state.stage == "body":
        points = (
            state.front_body_points
            if state.current_side == "front"
            else state.back_body_points
        )
        if len(points) == 2:
            points.clear()
            if state.current_side == "front":
                state.front_body_mask = None
            else:
                state.back_body_mask = None

        points.append([x, y])

        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            top_left = [min(x1, x2), min(y1, y2)]
            bottom_right = [max(x1, x2), max(y1, y2)]
            points[0] = top_left
            points[1] = bottom_right
            box = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
            masks = predict_sam2(box)
            if state.current_side == "front":
                state.front_body_mask = masks
            else:
                state.back_body_mask = masks

        return get_current_mask_preview(), get_burn_radio_choices()

    else:
        current_pts = get_burn_current_points()
        groups = get_burn_groups()

        if len(current_pts) == 2:
            current_pts.clear()

        current_pts.append([x, y])

        if len(current_pts) == 2:
            x1, y1 = current_pts[0]
            x2, y2 = current_pts[1]
            top_left = [min(x1, x2), min(y1, y2)]
            bottom_right = [max(x1, x2), max(y1, y2)]
            box = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
            masks = predict_sam2(box)
            groups.append(
                {"points": [list(top_left), list(bottom_right)], "mask": masks}
            )
            state.selected_burn_idx = len(groups) - 1
            current_pts.clear()

        return get_current_mask_preview(), get_burn_radio_choices()


def mask_bounding_box(mask):
    """Compute bounding box from a binary mask. Returns (y1, y2, x1, x2)."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(y1), int(y2), int(x1), int(x2)


def pad_to_square(mask):
    """Pad mask to minimum square, keeping content centered."""
    h, w = mask.shape
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    padded = np.zeros((size, size), dtype=mask.dtype)
    padded[pad_h : pad_h + h, pad_w : pad_w + w] = mask
    return padded


def calculate_tbsa():
    if state.front_body_mask is None or state.back_body_mask is None:
        return "Please complete body segmentation first", ""

    fb_mask = state.front_body_mask[0] > 0
    bb_mask = state.back_body_mask[0] > 0

    # Front burn: empty mask if no groups
    front_burn_combined = np.zeros_like(fb_mask)
    for g in state.front_burn_groups:
        if g["mask"] is not None:
            front_burn_combined = np.logical_or(front_burn_combined, g["mask"][0] > 0)

    # Back burn: empty mask if no groups
    back_burn_combined = np.zeros_like(bb_mask)
    for g in state.back_burn_groups:
        if g["mask"] is not None:
            back_burn_combined = np.logical_or(back_burn_combined, g["mask"][0] > 0)

    # Crop using body mask bounding boxes
    fy1, fy2, fx1, fx2 = mask_bounding_box(fb_mask)
    fb_cropped = fb_mask[fy1 : fy2 + 1, fx1 : fx2 + 1]
    fburn_cropped = front_burn_combined[fy1 : fy2 + 1, fx1 : fx2 + 1]

    by1, by2, bx1, bx2 = mask_bounding_box(bb_mask)
    bb_cropped = bb_mask[by1 : by2 + 1, bx1 : bx2 + 1]
    bburn_cropped = back_burn_combined[by1 : by2 + 1, bx1 : bx2 + 1]

    # Pad to minimum square, keeping content centered
    fb_sq = pad_to_square(fb_cropped)
    bb_sq = pad_to_square(bb_cropped)
    fburn_sq = pad_to_square(fburn_cropped)
    bburn_sq = pad_to_square(bburn_cropped)

    fb = Image.fromarray((fb_sq * 255).astype(np.uint8))
    bb = Image.fromarray((bb_sq * 255).astype(np.uint8))
    fburn = Image.fromarray((fburn_sq * 255).astype(np.uint8))
    bburn = Image.fromarray((bburn_sq * 255).astype(np.uint8))

    # Area ratio method (on cropped masks, before padding)
    fburn_intersect = np.sum(fburn_cropped & fb_cropped)
    bburn_intersect = np.sum(bburn_cropped & bb_cropped)
    total_body = np.sum(fb_cropped) + np.sum(bb_cropped)
    area_ratio_pct = (
        (fburn_intersect + bburn_intersect) / total_body * 100
        if total_body > 0
        else 0.0
    )

    # BurnAreaNet method
    with torch.inference_mode():
        tbsa = burn_net(
            transform(fb).to(device).unsqueeze(0),
            transform(bb).to(device).unsqueeze(0),
            transform(fburn).to(device).unsqueeze(0),
            transform(bburn).to(device).unsqueeze(0),
        )
    burn_net_pct = tbsa.item() * 100

    return f"{burn_net_pct:.2f}%", f"{area_ratio_pct:.2f}%"


with gr.Blocks(title="Burn TBSA Calculator") as demo:
    gr.Markdown("# Burn TBSA Calculator")
    gr.Markdown(
        "Interactive segmentation using SAM2, followed by TBSA estimation using BurnAreaNet"
    )

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("## Interactive Segmentation")
            image_display = gr.Image(
                label="Click image to add points for segmentation",
                interactive=True,
                type="numpy",
                height=500,
            )
            with gr.Group(visible=False) as burn_panel:
                gr.Markdown("### Burn Region Management")
                burn_region_radio = gr.Radio(
                    choices=[],
                    label="Burn Areas (select to show box)",
                    interactive=True,
                )
                delete_burn_btn = gr.Button("Delete Selected Burn Area", variant="stop")

        with gr.Column(scale=2):
            gr.Markdown("## 1. Load Images")
            with gr.Row():
                front_upload = gr.File(label="Front Image", file_types=["image"])
                back_upload = gr.File(label="Back Image", file_types=["image"])
            with gr.Row():
                load_btn = gr.Button("Load Images", variant="primary")
                load_examples_btn = gr.Button("Load Examples")

            gr.Markdown("## 2. Select Parameters")
            side_radio = gr.Radio(
                ["front", "back"], label="Select Image", value="front"
            )
            stage_radio = gr.Radio(["body", "burn"], label="Select Stage", value="body")

            gr.Markdown("## 3. TBSA Calculation")
            tbsa_btn = gr.Button("Calculate TBSA", variant="primary", size="lg")
            tbsa_burn_net_result = gr.Textbox(label="BurnAreaNet TBSA", value="")
            tbsa_area_ratio_result = gr.Textbox(label="Area Ratio TBSA", value="")

    load_btn.click(
        load_images,
        inputs=[front_upload, back_upload],
        outputs=[image_display, burn_region_radio],
    )
    load_examples_btn.click(
        load_example_images,
        outputs=[front_upload, back_upload, image_display, burn_region_radio],
    )
    side_radio.change(
        set_current_side,
        inputs=[side_radio],
        outputs=[image_display, burn_region_radio],
    )
    stage_radio.change(
        set_stage,
        inputs=[stage_radio],
        outputs=[image_display, burn_region_radio, burn_panel],
    )
    image_display.select(add_point, outputs=[image_display, burn_region_radio])
    burn_region_radio.select(select_burn_region, outputs=[image_display])
    delete_burn_btn.click(
        delete_burn_region,
        inputs=[burn_region_radio],
        outputs=[image_display, burn_region_radio],
    )
    tbsa_btn.click(
        calculate_tbsa,
        outputs=[tbsa_burn_net_result, tbsa_area_ratio_result],
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
