# Import necessary libraries
from tensorflow.keras.models import model_from_json 
import cv2
import numpy as np

# Load pre-trained model from JSON file and weights from H5 file
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model_weights.h5")

# Load pre-trained Haar Cascade classifier for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
# Define emotions as a tuple
emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

# Define window size and emotion threshold
window_size = 6
emotion_threshold = 0.1

def detect_emotion(img):
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using Haar Cascade classifier
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    
    # Initialize emotions_detected list
    emotions_detected = []

    # Loop over all detected faces
    for (x,y,w,h) in faces_detected:
        # Draw rectangle around detected face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=3)
        
        # Extract region of interest from the face and resize to 48x48
        roi_gray = gray_img[y:y+w,x:x+h]
        roi_gray = cv2.resize(roi_gray,(48,48))
        
        # Reshape the image to be compatible with the model
        img = roi_gray.reshape((1,48,48,1))
        
        # Normalize the pixel values
        img = img / 255.0

        # Get predicted emotion scores from the model
        emotion_scores = model.predict(img.reshape((1,48,48,1)))[0]

        # Keep only emotions with confidence score above threshold
        detected_emotions = [emotion for i, emotion in enumerate(emotions) if emotion_scores[i] > emotion_threshold]
        emotions_detected.extend(detected_emotions)

        # Print the detected emotions and their corresponding scores
        detected_emotions_scores = [(emotions[i], emotion_scores[i]) for i in range(len(emotions)) if emotion_scores[i] > emotion_threshold]
        for emotion, score in detected_emotions_scores:
            print(emotion, score*100)

    return emotions_detected


# Define a function that detects emotions in a video stream and maps them to a label
def video_detect(video_link = 0):

    # Initialize video capture object with the specified video link
    cap = cv2.VideoCapture(video_link)

    # Initialize a window to keep track of past emotions detected
    window = []

    # Loop over all frames in the video stream
    while True:

        # Read the next frame from the video stream
        ret, test_img = cap.read()

        # If there are no more frames, continue to the next iteration
        if not ret:
            continue

        # Detect emotions in the current frame
        emotions_detected = detect_emotion(test_img)

        # Add the detected emotions to the window
        window.append(emotions_detected)

        # If the window is not full yet, continue
        if len(window) < window_size:
            continue
        # If the window is too full, remove the oldest element
        elif len(window) > window_size:
            window.pop(0)

        # Calculate the mean confidence scores for each emotion over the window
        mean_confidences = []
        for i in range(len(emotions)):
            emotion_confidences = [frame.count(emotions[i]) / window_size for frame in window]
            mean_confidences.append(np.mean(emotion_confidences))

        # Get the three highest-scoring emotions as the emotion pattern
        sorted_indices = np.argsort(mean_confidences)[::-1]
        E1, E2 = [emotions[i] for i in sorted_indices[:2]]
        EP = (E1, E2)

        # Map the emotion pattern to a label
        if EP == ('Neutral', 'Surprise'):
            label = 'Confusion'
        elif EP == ('Happy', 'Neutral'):
            label = 'Satisfaction/Delighted'
        elif EP == ('Neutral', 'Sad'):
            label = 'Disappointment/Dissatisfaction'
        elif set(EP) == set(('Sad', 'Angry')):
            label = 'Frustration'

        # Add the label to the frame and display it
        cv2.putText(test_img, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis', resized_img)

        # If the user presses 's', stop the loop and exit the program
        if cv2.waitKey(10) == ord('s'):
            break

    # Release the video capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
 
video_detect()
