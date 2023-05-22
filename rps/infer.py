import cv2
from PIL import Image
from .predict_image import predict_image

def live_inference(model, transform, class_names, frame_size=(400, 400)):
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define the region of interest (ROI) for the gesture
        height, width, _ = frame.shape
        x1, y1 = (width // 2) - (frame_size[0] // 2) - 300, (height // 2) - (frame_size[1] // 2) - 100
        x2, y2 = x1 + frame_size[0], y1 + frame_size[1]

        # Extract the ROI
        roi = frame[y1:y2, x1:x2]

        # Convert the ROI to PIL Image
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        # # Make a prediction
        prediction, probability = predict_image(pil_img, model, transform)
        predicted_class = class_names[prediction]

        # Draw the highlighted frame on the original frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # # Display the prediction on the frame
        cv2.putText(
            img=frame, 
            text=f'{predicted_class}: {probability:.1%}', 
            org=(10, 30), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale = 1, 
            color=(0, 255, 0), 
            thickness=2
            )

        # Show the frame
        cv2.imshow("Live Inference", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()