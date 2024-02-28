import cv2
import io
import requests
import uuid
import time
import json
from minio import Minio
import subprocess
import re
import platform

def get_network_ip():
    """
    Retrieves the local network IP address of the system.

    This function works on both Windows and Unix-like systems (Linux, macOS).
    On Windows, it runs the 'ipconfig' command to extract the IPv4 address.
    On Unix-like systems, it runs the 'ifconfig' command to extract the IPv4 address.

    Returns:
        str or None: The IPv4 address of the local network interface, or None if the address couldn't be determined.
    """
    try:
        system = platform.system()
        if system == "Windows":
            result = subprocess.run(['ipconfig'], capture_output=True, text=True)
            config_output = result.stdout
            # Regular expression to find the IPv4 address
            ip_pattern = r'IPv4 Address[^\n]*:\s*([^\s]+)'
            match = re.search(ip_pattern, config_output)
            if match:
                return match.group(1)
            else:
                return None
        elif system == "Linux" or system == "Darwin":  # Unix-like systems
            result = subprocess.run(['ifconfig'], capture_output=True, text=True)
            ifconfig_output = result.stdout
            # Regular expression to find the IPv4 address
            ip_pattern = r'inet (\d+\.\d+\.\d+\.\d+)'
            match = re.search(ip_pattern, ifconfig_output)
            if match:
                return match.group(1)
            else:
                return None
        else:
            print("Unsupported operating system")
            return None
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def process_frame(frame, frame_count:int):
    """
    Preprocesses a single frame of the video stream and downsamples it to 480p.
    Args:
        frame (numpy.ndarray): The frame to be processed.
        frame_count (int): The index of the frame.
    Returns:
        tuple: A tuple containing the filename and processed frame image bytes.
    """
    try:
        if frame is not None and frame.size != 0:
            # Downsample the frame to 480p resolution
            frame = cv2.resize(frame, (640, 480))  # Adjust the dimensions as needed

            filename = f"frame_{frame_count:05d}.jpg"
            image_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[
                1
            ].tobytes()
            return filename, image_bytes
        else:
            return None
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def video_ingestion(url:str):
    """
    Ingests a video stream from the given URL, preprocesses its frames,
    and saves the processed frames to MinIO.
    Args:
        url (str): The URL of the video stream.
    Returns:
        str: The name of the MinIO bucket where the processed frames are saved.
    """
    try:
        
        # Fetch the video data from the URL
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print("Failed to fetch video stream")
            return None

        # Read the video stream from the URL
        video_stream = cv2.VideoCapture(url)
        if not video_stream.isOpened():
            print("Failed to open video stream")
            return None

        # Get the video frame rate and total number of frames
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize the frame counter and elapsed time
        frame_count = 0
        elapsed_time = 0.0
        start_time = time.time()

        # Loop through the frames of the video stream
        output_data = []
        while True:
            # Read the next frame from the video stream
            ret, frame = video_stream.read()
            if not ret:
                break

            # Process the frame
            result = process_frame(frame, frame_count)
            if result is not None:
                output_data.append(result)

            # Print the current minute of the video
            frame_count += 1
            elapsed_time = frame_count / fps
            if frame_count % (fps * 60) == 0:
                minutes = int(elapsed_time / 60)
                seconds = int(elapsed_time % 60)
                print(
                    f"Processed {minutes:02d}:{seconds:02d} ({frame_count}/{total_frames} frames)"
                )

        # Save the processed frames to MinIO
        bucket_name = str(uuid.uuid4())
        folder_name = "ingestion"

        # Change credentials accordingly or use keyvaults.
        minio_client = Minio(
            f"{get_network_ip()}:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False,
        )

        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        object_prefix = f"{folder_name}/"

        for i, (filename, result) in enumerate(output_data):
            object_name = f"{object_prefix}{filename}"

            # Save the processed frame to MinIO
            with io.BytesIO(result) as f:
                minio_client.put_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    data=f,
                    length=len(result),
                    content_type="image/jpeg",
                )

        minio_client.__del__()

        print(f"Video ingestion succeeded on bucket: {bucket_name}")

        return bucket_name
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def invoke_object_detection(bucket_name:str, object_detection:str, probability:str):
    """
    Invokes the object detection OpenFaaS function on the frames in the specified bucket.
    Args:
        bucket_name (str): The name of the MinIO bucket.
        object_detection (str): The object to search for in the stream.
        probability (str): Min. required probability to classify an object as the requested one.
    Returns:
        None
    """
    try:
        # Define the URL of the object detection OpenFaaS function
        url = f"{get_network_ip()}:8080/function/object-detection"

        # Create the payload data
        payload = {
            "bucket_name": bucket_name,
            "objectDetection": object_detection,
            "probability": probability,
        }

        print(payload)

        # Send a POST request to the object detection function
        response = requests.post(url, json=payload)

        # Check the response status code
        if response.status_code == 200:
            print("Object detection function invoked successfully")
        else:
            print("Object detection function invocation failed")

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

    
def handle(req):
    """
    Entry point for the serverless function.
    
    Args:
        req (str): The HTTP request body.
        
    Returns:
        str or bool: The result of the function execution.
    """
    try:
        start_time = time.time()
    
        # Parse the request body to get the bucket name and URL
        try:
            req_data = json.loads(req)
            url = req_data.get("url")
            object_detection = req_data.get("objectDetection")
            probability = req_data.get("probability")
        except json.JSONDecodeError:
            print("Invalid request body")
            return False

        # Call the video ingestion function with the provided URL
        if url:
            bucket_name = video_ingestion(url)
            if bucket_name:
                elapsed_time_ingestion = time.time() - start_time
                print(f"Video ingestion time: {elapsed_time_ingestion:.2f} seconds")

                start_time_detection = time.time()
                invoke_object_detection(bucket_name, object_detection, probability)
                elapsed_time_detection = time.time() - start_time_detection
                print(f"Object detection time: {elapsed_time_detection:.2f} seconds")
                return bucket_name
            else:
                return False
        else:
            print("URL not provided")
            return False
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

