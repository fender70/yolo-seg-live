import cv2
from ultralytics import YOLO

def run_live_segmentation(model_path='trash-detector-1.pt', source=0):
    """
    Performs live instance segmentation using a YOLO model.

    Args:
        model_path (str): Path to the YOLO segmentation model weights (e.g., 'yolov8n-seg.pt').
        source (int or str): Video source. 0 for webcam, or a path to a video file.
    """
    # Load the YOLO segmentation model
    model = YOLO("./trash-detector-1.pt")

    # Open the video source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source}.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from video source.")
            break

        # Perform inference on the frame
        results = model(frame, stream=True, conf=0.5)  # stream=True for generator of results

        # Process and display results
        for r in results:
            annotated_frame = r.plot()  # Plot detections and masks on the frame

            # Display the annotated frame
            cv2.imshow("YOLO Segmentation Live Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage:
    # Use 'yolov8n-seg.pt' for a nano-sized segmentation model
    # Use 0 for the default webcam, or provide a video file path like 'path/to/your/video.mp4'
    run_live_segmentation(model_path='./trash-detector-1.pt', source=0)