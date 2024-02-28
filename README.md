
# Real-time Video Ingestion and Object Detection System

This system comprises two serverless functions deployed on OpenFaaS, leveraging MinIO for object storage, Python for scripting, YOLO for object detection, and OpenCV for image processing.

This code is a minified version of my thesis. You can find my thesis on [ResearchGate](https://www.researchgate.net/publication/377466295_Serverless_Video_Analytics_Platform) .

## Function 1: Video Ingestion and Object Detection Trigger

**Function 1** serves as the primary entry point, responsible for initiating video ingestion, preprocessing frames, storing them in MinIO, and triggering object detection.

### Purpose and Workflow:
- **Purpose:** Function 1 facilitates real-time video analysis for use cases such as surveillance, anomaly detection, and automated monitoring.
- **Workflow:**
  1. **Ingestion Initiation:** Receives a JSON payload containing the video stream URL, target object for detection, and probability threshold.
  2. **Stream Processing:** Fetches the video stream, extracts and preprocesses frames, ensuring compatibility with downstream processing.
  3. **Data Storage:** Stores preprocessed frames in MinIO, ensuring accessibility and scalability for subsequent analysis.
  4. **Action Invocation:** Triggers Function 2 (Object Detection) to perform object detection on stored frames.

## Function 2: Object Detection

**Function 2** specializes in detecting objects within images stored in MinIO, annotating them with bounding boxes for visualization and further analysis.

### Purpose and Workflow:
- **Purpose:** Function 2 provides robust object detection capabilities, essential for tasks like inventory management, security monitoring, and content moderation.
- **Workflow:**
  1. **Data Retrieval:** Retrieves images from the designated MinIO bucket, processing them based on provided criteria.
  2. **Object Detection:** Utilizes YOLO to detect objects within images, applying bounding boxes to highlight their presence.
  3. **Annotation:** Annotates detected objects within images, enhancing their interpretability and facilitating downstream processing.
  4. **Data Persistence:** Stores annotated images back in the MinIO bucket, ensuring the preservation of analysis results.

## Deployment Instructions:

This section provides instructions for deploying the system, including setup, configuration, and invocation steps.

# Real-time Video Ingestion and Object Detection System

This advanced system comprises two serverless functions deployed on OpenFaaS, leveraging MinIO for object storage, Python for scripting, YOLO for object detection, and OpenCV for image processing.

## Deployment Instructions:

To deploy the system, follow these steps:

1. **Set Up MinIO and OpenFaaS:**
   - Ensure MinIO and OpenFaaS are running in Docker containers.
   - Configure the connection between MinIO and OpenFaaS to enable seamless data transfer.

2. **Build the Functions:**
   - Use the `faas-cli build` command to build the functions based on the provided YAML configuration file (`function.yml`). 
     ```bash
     faas-cli build -f function.yml
     ```

3. **Deploy the Functions:**
   - Deploy the built functions using the `faas-cli deploy` command.
     ```bash
     faas-cli deploy -f function.yml
     ```

4. **Configure MinIO Credentials:**
   - Ensure that the MinIO access key and secret key are correctly configured within the function code or provided via environment variables.

5. **Verify Deployment:**
   - Once deployed, verify the successful deployment of the functions and their availability for invocation.
   - Ensure that the MinIO bucket specified in the function code exists and is accessible.

6. **Invoke the Functions:**
   - Trigger Function 1 (Video Ingestion and Object Detection Trigger) by sending a JSON payload containing the necessary parameters (video stream URL[HTTP or RTSP], object to detect, probability threshold).
   - Monitor the system logs to track the progress of video ingestion, object detection, and any potential errors or issues.

7. **Enjoy fast Video Analysis in a Serverless Manner:**
   - Sit back, relax, and witness the magic of real-time video ingestion and object detection as the system processes incoming streams and identifies objects of interest with remarkable accuracy.

## Additional Notes:
- **Artifact Requirements:** Ensure the availability of YOLO configuration files (`yolov3-tiny.weights`, `yolov3-tiny.cfg`, `coco.names`) in the designated paths relative to function containers.
- **Networking Considerations:** Ensure proper network configuration to enable seamless communication between the MinIO server and function instances.



## Additional Notes:

- **Artifact Requirements:** Ensure the availability of YOLO configuration files (`yolov3-tiny.weights`, `yolov3-tiny.cfg`, `coco.names`) in the designated paths relative to function containers.
- **Networking Considerations:** Ensure proper network configuration to enable seamless communication between the MinIO server and function instances.

