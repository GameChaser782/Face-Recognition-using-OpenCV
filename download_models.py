import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

def download_file(url, save_path):
    """Download a file from a URL to a specific path"""
    print(f"Downloading {url} to {save_path}...")
    urllib.request.urlretrieve(url, save_path)
    print(f"Downloaded {save_path}")

def main():
    # Create Models directory if it doesn't exist
    models_dir = Path("Models")
    models_dir.mkdir(exist_ok=True)
    
    # Create a temporary directory for downloads
    temp_dir = Path("temp_downloads")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Download face detection models
        print("Downloading face detection models...")
        face_detection_models = [
            {
                "url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
                "filename": "haarcascade_frontalface_default.xml"
            },
            {
                "url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml",
                "filename": "haarcascade_frontalface_alt.xml"
            },
            {
                "url": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/opencv_face_detector.pbtxt",
                "filename": "opencv_face_detector.pbtxt"
            }
        ]
        
        for model in face_detection_models:
            download_file(model["url"], models_dir / model["filename"])
        
        # Download face detection frozen model
        face_detection_pb_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb"
        download_file(face_detection_pb_url, models_dir / "opencv_face_detector_uint8.pb")
        
        # Download age and gender model prototxt files
        print("Downloading age and gender model prototxt files...")
        prototxt_files = [
            {
                "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy_age.prototxt",
                "filename": "deploy_age.prototxt"
            },
            {
                "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy_gender.prototxt",
                "filename": "deploy_gender.prototxt"
            }
        ]
        
        for model in prototxt_files:
            download_file(model["url"], models_dir / model["filename"])
        
        # Download Caffe model files (these are large files)
        print("Downloading age and gender Caffe model files (this might take a while)...")
        print("Note: These are large files. If download fails, please download manually from the links in the README.")
        
        # Age model
        age_model_url = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel"
        download_file(age_model_url, models_dir / "age_net.caffemodel")
        
        # Gender model
        gender_model_url = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel"
        download_file(gender_model_url, models_dir / "gender_net.caffemodel")
        
        print("\nAll models downloaded successfully!")
        print("The models are saved in the 'Models' directory.")
        
    except Exception as e:
        print(f"Error downloading models: {e}")
        print("\nPlease download the models manually as described in the README.md file.")
    
    finally:
        # Clean up the temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    print("Model Downloader for Face Recognition App")
    print("=========================================")
    choice = input("This will download approximately 90MB of model files. Continue? (y/n): ")
    if choice.lower() == 'y':
        main()
    else:
        print("Download cancelled.") 