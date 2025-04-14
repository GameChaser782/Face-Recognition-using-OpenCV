import streamlit as st
import numpy as np
import cv2
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Face Recognition App",
    page_icon="🧑",
    layout="wide"
)

# Constants and paths
MODELS_DIR = Path("Models")  # Path to models directory

# Cache model loading to avoid reloading on every rerun
@st.cache_resource
def load_face_recognition_models():
    # Load face cascade for KNN-based recognition
    face_cascade = cv2.CascadeClassifier(str(MODELS_DIR / "haarcascade_frontalface_alt.xml"))
    if face_cascade.empty():
        # Try default model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return face_cascade

@st.cache_resource
def load_age_gender_models():
    # Load models for age and gender detection
    face_proto = str(MODELS_DIR / "opencv_face_detector.pbtxt")
    face_model = str(MODELS_DIR / "opencv_face_detector_uint8.pb")
    age_proto = str(MODELS_DIR / "deploy_age.prototxt")
    age_model = str(MODELS_DIR / "age_net.caffemodel")
    gender_proto = str(MODELS_DIR / "deploy_gender.prototxt")
    gender_model = str(MODELS_DIR / "gender_net.caffemodel")
    
    # Load networks
    face_net = cv2.dnn.readNet(face_model, face_proto)
    age_net = cv2.dnn.readNet(age_model, age_proto)
    gender_net = cv2.dnn.readNet(gender_model, gender_proto)
    
    return face_net, age_net, gender_net

# Helper Functions for Age & Gender Detection
def detect_face_age_gender(face_net, frame, confidence_threshold=0.7):
    frame_copy = frame.copy()
    frame_height = frame_copy.shape[0]
    frame_width = frame_copy.shape[1]
    
    # Create a 4D blob from frame
    blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
    
    # Pass the blob through the network and get detections
    face_net.setInput(blob)
    detections = face_net.forward()
    
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height/150)), 8)
    
    return frame_copy, face_boxes

def predict_age_gender(frame, face_boxes, age_net, gender_net, padding=20):
    gender_list = ['Male', 'Female']
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    
    result_img = frame.copy()
    
    for face_box in face_boxes:
        x1, y1, x2, y2 = face_box
        # Extract face
        face = frame[max(0, y1-padding):min(y2+padding, frame.shape[0]-1), 
                    max(0, x1-padding):min(x2+padding, frame.shape[1]-1)]
        
        # Create blob from face
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
        
        # Gender prediction
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        
        # Age prediction
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        
        # Display the result
        label = f"{gender}, {age}"
        cv2.putText(result_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return result_img

# Helper Functions for KNN-based Face Recognition
def distance(v1, v2):
    # Euclidean distance
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
    dist = []
    
    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]
    
    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]

def detect_and_recognize_faces(frame, face_cascade, trainset, names):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect multiple faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    result_img = frame.copy()
    
    for face in faces:
        x, y, w, h = face
        
        # Get the face region of interest
        offset = 5
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        try:
            face_section = cv2.resize(face_section, (100, 100))
            
            # Use KNN to predict the name
            out = knn(trainset, face_section.flatten())
            name = names.get(int(out), "Unknown")
            
            # Draw rectangle in the original image
            cv2.putText(result_img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 255, 255), 2)
        except Exception as e:
            st.error(f"Error processing face: {e}")
    
    return result_img

# Function to load face data for recognition
def load_face_data(dataset_path):
    face_data = []
    labels = []
    class_id = 0
    names = {}
    
    try:
        for fx in os.listdir(dataset_path):
            if fx.endswith('.npy'):
                names[class_id] = fx[:-4]
                data_item = np.load(os.path.join(dataset_path, fx))
                
                # Ensure consistent dimensions by flattening face samples
                if data_item.ndim == 4:  # If shape is (samples, height, width, channels)
                    samples = data_item.shape[0]
                    data_item = data_item.reshape(samples, -1)  # Flatten each face sample
                    
                face_data.append(data_item)
                
                target = class_id * np.ones((data_item.shape[0],))
                class_id += 1
                labels.append(target)
        
        if face_data:
            # Now all arrays should have consistent dimensions
            face_dataset = np.concatenate(face_data, axis=0)
            face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
            trainset = np.concatenate((face_dataset, face_labels), axis=1)
            return trainset, names
    except Exception as e:
        st.error(f"Error loading face data: {e}")
    
    # Return empty data if no faces found
    return np.array([]), {}

def collect_face_data(name):
    """Collect face data for training"""
    st.subheader(f"Collecting face data for: {name}")
    
    face_cascade = load_face_recognition_models()
    
    # Create directory if it doesn't exist
    dataset_path = Path("Models")
    dataset_path.mkdir(exist_ok=True)
    
    # For storing face data
    face_data = []
    count = 0
    max_samples = 50  # Number of face samples to collect
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    # Create placeholders for the webcam feed and a progress bar
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while count < max_samples:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame from camera")
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face
            offset = 10
            face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
            try:
                face_section = cv2.resize(face_section, (100, 100))
                
                # Add to face data every 10 frames
                if count % 10 == 0:
                    # Store flattened face data to ensure consistency
                    face_data.append(face_section.flatten())
                    
                count += 1
                progress_bar.progress(count / max_samples)
                status_text.text(f"Collecting samples: {count}/{max_samples}")
            except Exception as e:
                st.warning(f"Could not process face: {e}")
        
        # Display the frame - fix deprecated parameter
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)
        
        # Break if we've collected enough samples
        if count >= max_samples:
            break
            
        # Add a small delay
        cv2.waitKey(1)
    
    # Release webcam
    cap.release()
    
    if face_data:
        # Convert to numpy array and save - store as 2D array (samples, flattened_pixels)
        face_data = np.array(face_data)
        file_path = dataset_path / f"{name}.npy"
        np.save(str(file_path), face_data)
        st.success(f"Face data for {name} saved successfully!")
    else:
        st.error("No face data collected.")

# Main App UI
def main():
    st.title("Face Recognition App 👤")
    
    # Sidebar for app navigation
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["About", "Age & Gender Detection", "Face Recognition", "Add New Face"]
    )
    
    # About page
    if app_mode == "About":
        st.markdown("""
        ## About
        
        This application demonstrates face recognition capabilities using OpenCV and deep learning. 
        
        ### Features
        - **Age & Gender Detection**: Detect faces and predict age and gender
        - **Face Recognition**: Recognize known faces using a KNN algorithm
        - **Add New Face**: Add your face to the recognition database
        
        ### Implementation
        The app uses:
        - OpenCV for computer vision tasks
        - Deep neural networks for age and gender prediction
        - K-Nearest Neighbors (KNN) algorithm for face recognition
        - Streamlit for the user interface
        
        ### Note
        The app works best with good lighting and clear faces facing the camera.
        """)
    
    # Age & Gender Detection mode
    elif app_mode == "Age & Gender Detection":
        st.header("Age & Gender Detection")
        
        try:
            # Load models
            face_net, age_net, gender_net = load_age_gender_models()
            
            run = st.checkbox("Start Detection")
            
            if run:
                # Create a placeholder for webcam image
                video_placeholder = st.empty()
                
                # Start webcam
                cap = cv2.VideoCapture(0)
                
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to grab frame from webcam")
                        break
                    
                    # Detect faces
                    result_img, face_boxes = detect_face_age_gender(face_net, frame)
                    
                    # Predict age and gender for each face
                    if face_boxes:
                        result_img = predict_age_gender(frame, face_boxes, age_net, gender_net)
                    
                    # Display the result - fix deprecated parameter
                    video_placeholder.image(result_img, channels="BGR", use_container_width=True)
                
                # Release resources when stopped
                cap.release()
        except Exception as e:
            st.error(f"Error: {e}")
            st.warning("Make sure the model files are in the 'Models' directory.")
    
    # Face Recognition mode
    elif app_mode == "Face Recognition":
        st.header("Face Recognition")
        
        # Check and load models and face data
        face_cascade = load_face_recognition_models()
        
        # Load face data
        dataset_path = "Models"
        trainset, names = load_face_data(dataset_path)
        
        if len(names) == 0:
            st.warning("No face data found. Please add faces using the 'Add New Face' option.")
        else:
            st.success(f"Loaded {len(names)} faces: {', '.join(names.values())}")
            
            run = st.checkbox("Start Recognition")
            
            if run:
                # Create a placeholder for webcam image
                video_placeholder = st.empty()
                
                # Start webcam
                cap = cv2.VideoCapture(0)
                
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to grab frame from webcam")
                        break
                    
                    # Detect and recognize faces
                    result_img = detect_and_recognize_faces(frame, face_cascade, trainset, names)
                    
                    # Display the result - fix deprecated parameter
                    video_placeholder.image(result_img, channels="BGR", use_container_width=True)
                
                # Release resources when stopped
                cap.release()
    
    # Add New Face mode
    elif app_mode == "Add New Face":
        st.header("Add New Face")
        
        # Input for the name
        name = st.text_input("Enter the name for the face:")
        
        if name:
            if st.button("Start Collecting Face Data"):
                collect_face_data(name)

if __name__ == "__main__":
    main() 