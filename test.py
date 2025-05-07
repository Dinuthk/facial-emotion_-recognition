import cv2
import numpy as np
# from keras.models import load_model
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import os

# Load the pre-trained model
model = load_model('ai_project_model1_file.h5')

# Start video capture
video = cv2.VideoCapture(0)

# Load face detection classifier
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Dictionary mapping emotions to their corresponding label text (for console output)
labels_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

# Create image paths for emoji images
# You'll need to have these images in your working directory or specify the full path
emoji_files = {
    0: 'angry.jpg',     # Path to angry emoji image
    1: 'disgust.png',   # Path to disgust emoji image
    2: 'fear.png',      # Path to fear emoji image
    3: 'happy.png',     # Path to happy emoji image
    4: 'neutral.png',   # Path to neutral emoji image
    5: 'sad.png',       # Path to sad emoji image
    6: 'surprise.png'   # Path to surprise emoji image
}

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    
    for x, y, w, h in faces:
        # Extract and preprocess face region
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        
        # Make prediction
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        print(label)
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        
        # Print detected emotion label to console
        print(f"Detected: {labels_dict[label]}")
        
        # Check if emoji file exists
        emoji_path = emoji_files[label]
        if os.path.exists(emoji_path):
            try:
                # Load and resize emoji image
                emoji_img = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                
                # If the emoji has transparency (alpha channel)
                if emoji_img.shape[2] == 4:
                    # Resize emoji (adjust size as needed)
                    emoji_size = min(w, h) // 2
                    emoji_img = cv2.resize(emoji_img, (emoji_size, emoji_size))
                    
                    # Calculate position to place emoji (above face)
                    emoji_x = x + (w - emoji_size) // 2
                    emoji_y = y - emoji_size - 10
                    
                    # Make sure emoji is within frame
                    if emoji_y >= 0 and emoji_x >= 0 and emoji_x + emoji_size <= frame.shape[1]:
                        # Get alpha channel
                        alpha_channel = emoji_img[:, :, 3] / 255.0
                        alpha_channel = np.repeat(alpha_channel[:, :, np.newaxis], 3, axis=2)
                        
                        # Get region of interest in the frame
                        roi = frame[emoji_y:emoji_y+emoji_size, emoji_x:emoji_x+emoji_size]
                        
                        # Combine roi and emoji
                        foreground = emoji_img[:, :, :3] * alpha_channel
                        background = roi * (1 - alpha_channel)
                        result = foreground + background
                        
                        # Place emoji on frame
                        frame[emoji_y:emoji_y+emoji_size, emoji_x:emoji_x+emoji_size] = result
                else:
                    # If no alpha channel, resize and place directly
                    emoji_size = min(w, h) // 2
                    emoji_img = cv2.resize(emoji_img, (emoji_size, emoji_size))
                    emoji_x = x + (w - emoji_size) // 2
                    emoji_y = y - emoji_size - 10
                    
                    if emoji_y >= 0 and emoji_x >= 0 and emoji_x + emoji_size <= frame.shape[1]:
                        frame[emoji_y:emoji_y+emoji_size, emoji_x:emoji_x+emoji_size] = emoji_img
            except Exception as e:
                print(f"Error displaying emoji: {e}")
                # Fallback to text
                cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
                cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # Fallback to text if emoji file not found
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()