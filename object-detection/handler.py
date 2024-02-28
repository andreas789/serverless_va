import cv2
import numpy as np
from minio import Minio
from minio.error import S3Error
import io
import time
import json
import logging
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

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

class ObjectDetector:
    def __init__(self, minio_client, confidence_threshold):
        """
        Initializes the ObjectDetector.

        Parameters:
        - minio_client (Minio): Minio client for accessing the object storage.
        - confidence_threshold (float): Minimum confidence threshold for object detection.
        """
        try:
            self.minio_client = minio_client
            self.net = self.initialize_network()
            self.labels = self.load_labels()
            self.confidence_threshold = confidence_threshold
            self.frames_with_detections = 0
            self.total_frames = 0
        
        except Exception as e:
            print(f"Error occurred: {e}")
            return None
        

    def initialize_network(self):
        """
        Initializes the neural network for object detection.
        Path following is relative to the function once deployed on OpenFaas.
            -- > For locally testing you have to reassign it.

        Returns:
        - net (cv2.dnn_Net): Initialized neural network.
        """
        try:
            weights_path = "./function/data/yolov3-tiny.weights"
            config_path = "./function/data/yolov3-tiny.cfg"
            net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            return net
        
        except Exception as e:
            print(f"Error occurred: {e}")
            return None

    def load_labels(self):
        """
        Loads the object class labels from a file.

        Returns:
        - labels (list): List of object class labels.
        """
        try:
            labels_path = "./function/data/coco.names"

            with open(labels_path, "r") as f:
                labels = [line.strip() for line in f.readlines()]
            return labels

        except Exception as e:
            print(f"Error occurred: {e}")
            return None

    def detect_objects(self, image_data, object_classes):
        """
        Detects objects in an image.

        Parameters:
        - image_data (bytes): Image data in bytes.
        - object_classes (list): List of object classes to detect.

        Returns:
        - image (np.ndarray): Image with bounding boxes drawn around detected objects.
        - detections (dict): Dictionary containing detected objects and their properties.
        """
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            height, width, _ = image.shape
            blob = cv2.dnn.blobFromImage(
                image, 0.00392, (416, 416), swapRB=True, crop=False
            )

            self.net.setInput(blob)
            outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

            color_map = {
                "person": (0, 0, 255),  # Red for person
                "car": (0, 255, 0),  # Green for car
                # Add more object classes and corresponding colors here according to coco objects.
            }

            detections = {obj_class: [] for obj_class in object_classes}
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    obj_class = self.labels[class_id]

                    if (
                        confidence > self.confidence_threshold
                        and obj_class in object_classes
                    ):
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = center_x - w // 2
                        y = center_y - h // 2

                        color = color_map[obj_class]
                        detections[obj_class].append(
                            {
                                "class": obj_class,
                                "confidence": confidence,
                                "bbox": [x, y, w, h],
                            }
                        )
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            if any(detections.values()):
                self.frames_with_detections += 1
            self.total_frames += 1

            return image, detections
        
        except Exception as e:
            print(f"Error occurred: {e}")
            return None

def process_bucket(self, bucket_name:str, object_classes:str, img_selection=None):
    """
    Processes all images in a bucket for object detection.

    Parameters:
    - bucket_name (str): Name of the bucket containing images.
    - object_classes (list): List of object classes to detect.
    - img_selection (str, optional): Range of images to process in the format "start_index-end_index". Defaults to None.
    """
    try:
        logger.info(f"Processing bucket: {bucket_name}")

        start_idx = 0
        end_idx = float('inf')

        if img_selection:
            try:
                # Parse img_selection to get the range
                start_idx, end_idx = map(int, img_selection.split("-"))
            except ValueError:
                logger.error("Invalid img_selection format. Please provide the range in the format 'start_index-end_index'. Processing all images in the bucket instead.")

        objects = self.minio_client.list_objects(bucket_name, prefix="ingestion/")
        for obj in objects:
            object_name = obj.object_name
            object_index = int(object_name.split("_")[1].split(".")[0])

            # Check if the object's index is within the specified range or process all if img_selection is not provided
            if start_idx <= object_index <= end_idx:
                logger.info(f"Processing object: {object_name}")

                if object_name.endswith(".jpg"):
                    data = self.minio_client.get_object(bucket_name, object_name).read()

                    try:
                        image_with_detections, detections = self.detect_objects(
                            data, object_classes
                        )

                        logger.info(f"Detections: {detections}")

                        with io.BytesIO() as output:
                            encoded_image = cv2.imencode(".jpg", image_with_detections)[1]
                            output.write(encoded_image.tobytes())
                            output.seek(0)

                            self.minio_client.put_object(
                                bucket_name=bucket_name,
                                object_name=f"results/{object_name}",
                                data=output,
                                length=output.getbuffer().nbytes,
                                content_type="image/jpeg",
                            )

                    except (cv2.error, S3Error) as err:
                        logger.error(f"Error processing object: {object_name}. Error: {err}")

    except Exception as e:
        logger.error(f"An error occurred while processing the bucket: {e}")



def handle(req):
    try:
        
        # Initialize Minio client object
        minio_client = Minio(
            f"{get_network_ip()}:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )

        start_time = time.time()

        req = json.loads(req)
        bucket_name = req.get("bucket_name")
        confidence_threshold = req.get("objectDetection")
        img_selection = req.get("img-selection")  # Default is null if not provided
        object_classes = req.get("objectDetection")

        object_detector = ObjectDetector(minio_client, confidence_threshold)
        object_detector.process_bucket(bucket_name, object_classes, img_selection)

        elapsed_time_detection = time.time() - start_time
        print(f"Object detection time: {elapsed_time_detection:.2f} seconds")

    except Exception as e:
        print(f"Error occurred: {e}")
        return None