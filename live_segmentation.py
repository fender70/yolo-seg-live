# live_superclass_segmentation.py
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "./trash-detector-1.pt"

# Your dataset's class list (order must match the model if model.names not present)
DATASET_NAMES = [
    'aluminum', 'cardboard', 'egg shell', 'facemask', 'food wrapper',
    'fruit peels', 'glass bottle', 'left-over food', 'paper',
    'plastic bag', 'plastic bottle', 'plastic container', 'plastic sachet',
    'plastic straw', 'styrofoam containers', 'treeleaves', 'uht carton',
    'vegetable peels'
]

# Map each fine-grained class to a superclass
# Map each fine-grained class to a superclass (plastics -> recyclables)
CLASS_TO_SUPER = {
    # Recyclables
    'aluminum': 'recyclables',
    'cardboard': 'recyclables',
    'glass bottle': 'recyclables',
    'paper': 'recyclables',
    'uht carton': 'recyclables',
    'plastic bottle': 'recyclables',     # PET bottles
    'plastic container': 'recyclables',  # rigid PP/HDPE tubs, etc.
    'plastic bag': 'recyclables',        # films/LDPE set to recyclables per your policy
    'plastic sachet': 'recyclables',
    'plastic straw': 'recyclables',

    # Compost
    'egg shell': 'compost',
    'fruit peels': 'compost',
    'left-over food': 'compost',
    'treeleaves': 'compost',
    'vegetable peels': 'compost',

    # Waste (landfill)
    'facemask': 'waste',
    'food wrapper': 'waste',
    'styrofoam containers': 'waste'
}

# Colors per superclass (BGR)
SUPER_COLORS = {
    'recyclables': (0, 180, 0),   # green
    'compost': (0, 140, 255),     # orange
    'waste': (0, 0, 220)          # red
}


def draw_mask(frame, mask, color, alpha=0.35):
    """
    Blend a single binary mask onto the frame using a solid BGR color and alpha.
    Handles size mismatches by resizing the mask to the frame size.
    """
    # Ensure binary uint8 mask {0,1}
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)

    H, W = frame.shape[:2]
    if mask.shape[:2] != (H, W):
        # Resize mask to match frame
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)

    # Prepare colored overlay
    colored = np.empty_like(frame)
    colored[:] = color  # BGR

    # Alpha blend only where mask==1 (broadcast-safe)
    m = mask[:, :, None]
    blended = (frame.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha).astype(np.uint8)
    frame = np.where(m == 1, blended, frame)
    return frame


def put_label(frame, text, x1, y1, color, font_scale=0.5, thickness=1):
    """
    Draw a filled label box with text at the top-left of a detection.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    y1 = max(y1, th + 6)  # avoid going above the frame
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_legend(frame, counts, start=(10, 24)):
    """
    Draw a small legend with per-superclass counts.
    """
    x, y = start
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = 20
    for sup in ['recyclables', 'compost', 'waste']:
        c = counts.get(sup, 0)
        color = SUPER_COLORS.get(sup, (200, 200, 200))
        label = f"{sup}: {c}"
        cv2.putText(frame, label, (x, y), font, 0.6, color, 2, cv2.LINE_AA)
        y += line_h


def run_live_segmentation(model_path=MODEL_PATH, source=0, conf=0.5):
    """
    Live YOLO inference with superclass labeling (recyclables/compost/waste).
    Works with detect or seg models; masks are filled if present.
    """
    model = YOLO(model_path)

    # Choose class-name source
    model_names = getattr(model, 'names', None)
    if model_names:
        class_names = [model_names[i] for i in range(len(model_names))]
    else:
        class_names = DATASET_NAMES

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}.")
        return

    window_name = "YOLO Superclass (recyclables / compost / waste)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: Failed to read frame from video source.")
                break

            counts = {'recyclables': 0, 'compost': 0, 'waste': 0}

            # Inference; stream=True yields generator of results
            for r in model(frame, stream=True, conf=conf):
                # Boxes present?
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    cls = r.boxes.cls.int().cpu().numpy().tolist()
                    confs = r.boxes.conf.cpu().numpy().tolist()

                    # Masks (optional for seg models)
                    masks = None
                    if getattr(r, 'masks', None) is not None and r.masks is not None:
                        masks = r.masks.data.cpu().numpy()  # (N, h, w)

                    for i, box in enumerate(xyxy):
                        x1, y1, x2, y2 = box.astype(int).tolist()
                        class_id = cls[i]
                        score = confs[i]
                        base_name = class_names[class_id] if class_id < len(class_names) else f"id_{class_id}"
                        super_name = CLASS_TO_SUPER.get(base_name, 'waste')  # default bucket
                        counts[super_name] = counts.get(super_name, 0) + 1
                        color = SUPER_COLORS.get(super_name, (200, 200, 200))

                        # Draw mask if available
                        if masks is not None and i < len(masks):
                            frame = draw_mask(frame, (masks[i] > 0.5).astype(np.uint8), color, alpha=0.35)

                        # Draw box + label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{super_name} {score:.2f}"
                        put_label(frame, label, x1, y1, color)

            # Legend
            draw_legend(frame, counts, start=(10, 24))

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Live YOLO superclass segmentation (recyclables/compost/waste)")
    p.add_argument("--model", type=str, default=MODEL_PATH, help="Path to YOLO model weights (.pt)")
    p.add_argument("--source", default=0, help="0 for webcam, or path/URL to video")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_live_segmentation(model_path=args.model, source=args.source, conf=args.conf)
