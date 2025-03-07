import cv2
from ultralytics import YOLO

def predict_video(video_path, output_path, model_path, conf=0.25):
    """
    Run predictions on a video using YOLO and save the output video with predictions.

    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the output video with predictions.
        model_path (str): Path to the YOLO model (.pt file).
        conf (float): Confidence threshold for predictions (default: 0.25).
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video

    # Initialize the video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process video frame by frame
    while True:
        ret, frame = video.read()
        if not ret:
            break  # End of video

        # Run prediction on the frame
        results = model.predict(source=frame, conf=conf, save=False, show=False)

        # Overlay predictions on the frame
        annotated_frame = results[0].plot()  # Automatically draws boxes and labels

        # Write the annotated frame to the output video
        out.write(annotated_frame)

    # Release resources
    video.release()
    out.release()
    print(f"Prediction completed. Output saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    video_path = "traffic-videos/test-video1.mp4"  # Path to input video
    output_path = "output_video.mp4"  # Path to save output video
    model_path = "models/VehicleDetectionYolov11LModel.pt"  # Path to your YOLO model

    predict_video(video_path, output_path, model_path)
    
