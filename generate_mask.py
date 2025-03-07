import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Canvas, Button
from PIL import Image, ImageTk
import os

# Constants for fixed resolution
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720

# Global variables
video_paths = []
current_video_index = 0
frame = None
mask = None
drawing = False
curve_points = []
playing = False
cap = None
output_folder = "masks"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to open file dialog and select multiple videos
def load_videos():
    global video_paths, current_video_index
    video_paths = filedialog.askopenfilenames(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if not video_paths:
        return  
    current_video_index = 0
    load_video_by_index(current_video_index)

# Function to load video by index
def load_video_by_index(index):
    global frame, tk_img, curve_points, current_video_index, cap
    if index < 0 or index >= len(video_paths):
        return  
    
    current_video_index = index
    if cap:
        cap.release()
    
    cap = cv2.VideoCapture(video_paths[current_video_index])
    read_frame()
    curve_points = []

# Function to read a frame and update canvas
def read_frame():
    global cap, frame, tk_img, playing
    if not cap:
        return
    
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_frame(frame, CANVAS_WIDTH, CANVAS_HEIGHT)  # Resize to fit canvas
        display_frame(frame)
    
    if playing:
        root.after(30, read_frame)  # Call function again to continue playing

# Function to resize frame while maintaining aspect ratio
def resize_frame(image, width, height):
    h, w, _ = image.shape
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a black background and center the resized frame
    output = np.zeros((height, width, 3), dtype=np.uint8)
    y_offset = (height - new_h) // 2
    x_offset = (width - new_w) // 2
    output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return output

# Function to display frame on the Tkinter canvas
def display_frame(image):
    global tk_img, canvas_img
    image = Image.fromarray(image)
    tk_img = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas_img = canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

# Function to toggle play/pause
def toggle_play():
    global playing
    playing = not playing
    if playing:
        btn_play.config(text="⏸")
        read_frame()
    else:
        btn_play.config(text="▶️")

# Function to go to next video
def next_video():
    global current_video_index
    if current_video_index < len(video_paths) - 1:
        load_video_by_index(current_video_index + 1)

# Function to go to previous video
def prev_video():
    global current_video_index
    if current_video_index > 0:
        load_video_by_index(current_video_index - 1)

# Function to generate mask
def generate_mask():
    global mask, frame, current_video_index
    if frame is None or len(curve_points) < 3:
        return
    mask = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.uint8)
    pts = np.array(curve_points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    video_name = os.path.basename(video_paths[current_video_index])
    mask_filename = os.path.join(output_folder, os.path.splitext(video_name)[0] + "_mask.jpg")
    cv2.imwrite(mask_filename, mask)
    next_video()

# Function to clear selection
def clear_selection():
    global curve_points
    curve_points = []
    display_frame(frame)

# Mouse event functions
def on_mouse_press(event):
    global drawing, curve_points
    drawing = True
    curve_points = [(event.x, event.y)]

def on_mouse_move(event):
    global drawing, curve_points
    if drawing:
        curve_points.append((event.x, event.y))
        canvas.create_line(curve_points[-2][0], curve_points[-2][1], event.x, event.y, fill="red", width=2, smooth=True)

def on_mouse_release(event):
    global drawing
    drawing = False
    if len(curve_points) > 2:
        curve_points.append(curve_points[0])
        canvas.create_line(curve_points[-2][0], curve_points[-2][1], curve_points[-1][0], curve_points[-1][1], fill="red", width=2, smooth=True)

# Tkinter GUI setup
root = tk.Tk()
root.title("Smooth Curve Mask Selector")

# Canvas for video (fixed size)
canvas = Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="black")
canvas.pack()

# Button frame
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

# Buttons
btn_load = Button(btn_frame, text="📂 Load", command=load_videos)
btn_load.pack(side=tk.LEFT, padx=5)

btn_prev = Button(btn_frame, text="⏮", command=prev_video)
btn_prev.pack(side=tk.LEFT, padx=5)

btn_play = Button(btn_frame, text="▶️", command=toggle_play)
btn_play.pack(side=tk.LEFT, padx=5)

btn_next = Button(btn_frame, text="⏭", command=next_video)
btn_next.pack(side=tk.LEFT, padx=5)

btn_generate = Button(btn_frame, text="💾 Save Mask", command=generate_mask)
btn_generate.pack(side=tk.LEFT, padx=5)

btn_clear = Button(btn_frame, text="❌ Clear", command=clear_selection)
btn_clear.pack(side=tk.LEFT, padx=5)

# Mouse events
canvas.bind("<ButtonPress-1>", on_mouse_press)
canvas.bind("<B1-Motion>", on_mouse_move)
canvas.bind("<ButtonRelease-1>", on_mouse_release)

# Start Tkinter main loop
root.mainloop()