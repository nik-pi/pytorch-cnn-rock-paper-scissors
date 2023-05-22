import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from PIL import Image
import rps

st.title("Rock, paper, scissors live CNN")

model = rps.load_model()
transform = rps.data_transforms
class_names = ['paper', 'rock', 'scissors']

def callback(frame):
    frame = frame.to_ndarray(format="bgr24")
    frame_size = (200, 200)

    ###
    height, width, _ = frame.shape
    x1, y1 = (width // 2) - (frame_size[0] // 2) - 125, (height // 2) - (frame_size[1] // 2) - 80
    x2, y2 = x1 + frame_size[0], y1 + frame_size[1]

    # Extract the ROI
    roi = frame[y1:y2, x1:x2]

    # Convert the ROI to PIL Image
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    # # Make a prediction
    prediction, probability = rps.predict_image(pil_img, model, transform)
    predicted_class = class_names[prediction]

    # Draw the highlighted frame on the original frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display the prediction on the frame
    cv2.putText(
        img=frame, 
        text=predicted_class, 
        org=(50, 50), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale = 1, 
        color=(0, 255, 0), 
        thickness=2
        )


    ###


    return av.VideoFrame.from_ndarray(frame, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=callback)