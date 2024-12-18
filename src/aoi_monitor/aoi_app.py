#Imported Libraries 
import cv2 #OpenCv is for Image processing
import numpy as np #numpy used for working with arrays
import depthai as dai #DepthAI library is the communication with Oak device 
import blobconverter #converts neural networks to blob format to be deployed on devices that use Intel Myriad X VPU
import time #to time operations
import threading # allows proccesses to be run in parallel
from flask import Flask, Response, request, jsonify #for webserver functionality 
from flask_cors import CORS #handles cross-Origin Request for Flask Routes  


# Instantiate Flask app
app = Flask(__name__)
#CORS settings to app for  cross-origin requests
CORS(app)

# Function to encode the frame into JPEG format
def frame_to_jpeg(frame):
    _, jpeg = cv2.imencode('.jpg', frame)   
    return jpeg.tobytes()

#The entire class handles AI object detection
class AOIDetector:

    #constructor method
    def __init__(self):
        self.aoi_lock = threading.Lock() #Lock thread operations on AOI 
        self.create_pipeline() #pipeline is set up for image process
        self.run_detector()
    
    #sets up DepthAI pipline
    def create_pipeline(self):
        #AOI is defined and labels for detected objects 
        self.AOI = (50,50,150,150) #Initial AOI co-ordinates 
        
        #list of object labels that the neural network detects
        self.labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow","diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa" ,"train", "tvmonitor"]

        #a new pipeline is created
        self.pipeline = dai.Pipeline()

        # First, we want the Color camera as the output
        self.cam_rgb = self.pipeline.createColorCamera()
        self.cam_rgb.setPreviewSize(300, 300) #camera preview
        self.cam_rgb.setInterleaved(False)

        # setup neural network for detections 
        self.detection_nn = self.pipeline.createMobileNetDetectionNetwork()
        # Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
        # We're using a blobconverter tool to retreive the MobileNetSSD blob automatically from OpenVINO Model Zoo
        self.detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
        # Next, we filter out the detections that are below a confidence threshold. Confidence can be anywhere between <0..1>
        self.detection_nn.setConfidenceThreshold(0.5)
        # link camera to neural network input
        self.cam_rgb.preview.link(self.detection_nn.input)

        #set up XLinkOut for streaming neural network results to the host
        self.xout_rgb = self.pipeline.createXLinkOut()
        # For the rgb camera output, we want the XLink stream to be named "rgb"
        self.xout_rgb.setStreamName("rgb")
        # Linking camera preview to XLink input, so that the frames will be sent to host
        self.cam_rgb.preview.link(self.xout_rgb.input)
        
        # setup XLinkOut for streaming neural network input
        self.xout_nn = self.pipeline.createXLinkOut()
        #set stream name for neural network 
        self.xout_nn.setStreamName("nn")
        #link output of neural network to input of XLinkOut node 
        self.detection_nn.out.link(self.xout_nn.input)
    
    def run_detector(self):
        #initializing the OAK device with the pipeline 
        self.device = dai.Device(self.pipeline)

        #two queues for camera images and neural network results
        self.q_rgb = self.device.getOutputQueue("rgb")
        self.q_nn = self.device.getOutputQueue("nn")
         
    #Normalizes and clips bbox coordinates
    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
    
    #def getAOI(self):
      #  return self.AOI
    
    #this function update the AOI(Area of Interest)
    def setAOI(self, AOI):
        # Ensure thread safety when updating AOI
        with self.aoi_lock:
            try:
                # Assuming AOI is a dictionary with keys 'x_axis', 'y_axis', 'width', 'height'
                x = int(AOI['x_axis'])
                y = int(AOI['y_axis'])
                width = int(AOI['width'])
                height = int(AOI['height'])
                # Update the AOI tuple with the new values
                self.AOI = (x, y, width, height)

                # Log the updated AOI for verification
                print(f"Updated AOI: {self.AOI}")
            except KeyError as e:
                print(f"Missing key in AOI data: {e}")
            except ValueError as e:
                print(f"Invalid type for AOI data: {e}")


    #this function checks if there's objects detected within the Area of Interest(AOI)
    def detectionInAOI(self, AOI, bbox):
        #unpacking AOI and bounding box co-ordinates
        x_axis, y_axis, width, height = AOI

        EdgeR = x_axis + width # calculates the right edge of AOI
        EdgeL = y_axis + height# calculate the bottom edge of AOI 
        bbx_x1, bbx_y1, bbx_x2, bbx_y2 = bbox #Coordiantes of detected objects

    
       #This condition is basically checking if the bounding box is within the AOI, returns "True" if the bbox interwines with the AOI and if it doesn't return "False".
        return not (bbx_x2 < x_axis or  # BBox right edge is left of the AOI left edge
                 bbx_x1 > EdgeR or  # BBox left edge is right of the AOI right edge
                bbx_y2 < y_axis or  # BBox bottom edge is above the AOI top edge
                  bbx_y1 > EdgeL)  # BBox top edge is below the AOI bottom edge


      
    #Function to continuously yield frames for streaming 
    def get_next_frame(self):
        # Method to yield frames for streaming

        #infinite loop to keep checking for new frames  
        while True:
          in_rgb = self.q_rgb.tryGet()  #Attempt to get RGB frame from the queue
          in_nn = self.q_nn.tryGet() #Attempt to get neural network output from the queue

          frame = None #Initialize frame variable 
          detections = [] #List to store detections

          if in_rgb is not None:  #If an RGB frame is available
             frame = in_rgb.getCvFrame() #convert to OpenCV format

          if in_nn is not None: #if neural network output is available
                detections = in_nn.detections # Store the detections 

          aoi_color = (0, 255, 0)  # Initialize AOI color as green (no detection within AOI)

          if frame is not None: #If there's a frame to process
                for detection in detections: #Iterate through detections 
                    #Normalize and clip bounding box coordinates to frame size
                    bbox = self.frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    #Check if the detected object is a person and within the AOI
                    if self.labelMap[detection.label] == "person" and self.detectionInAOI(self.AOI, bbox):
                        #If there detection is found in the AOI, the color of the AOI, should change color to red. 
                        aoi_color = (0, 0, 255)
                        break  # Found a detection within AOI, no need to check further

                # Draw the AOI rectangle with the determined color
                top_left = (int(self.AOI[0]), int(self.AOI[1]))
                bottom_right = (int(self.AOI[0] + self.AOI[2]), int(self.AOI[1] + self.AOI[3]))
                cv2.rectangle(frame, top_left, bottom_right, aoi_color, 2)

                # Then draw bounding boxes for all detections on the frame
                for detection in detections:
                    bbox = self.frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

                # Display the frame
                cv2.imshow("preview", frame)
                
                #Encode the frame as JPEG 
                ret, buffer = cv2.imencode('.jpg', frame)  
                if not ret:
                  continue  # If encoding failed, skip this frame

                frame_bytes = buffer.tobytes() #Convert encoded frame to bytes 

            # Yield a multipart response containing the frame
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # If no frame is available, yield a placeholder or sleep briefly
             
 

          if cv2.waitKey(1) == ord('q'):
              break
    

#Attach an instance of the AOIDetector class to Flask app for all-round access
app.detector = AOIDetector()



# Other imports and code for your application setup...


# The rest of your Flask app...


#route for streaming video. This endpoint serves as a live video feed
@app.route('/video_stream')
def video_stream():
   return Response(app.detector.get_next_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

#route to update the Area of Interest through a POST request
@app.route('/updtAOI', methods=['POST'])
def updt_AOI():
    data = request.get_json()  # or request.json
    if data and 'AOI' in data:
        try:
            app.detector.setAOI(data['AOI'])
            return jsonify({'message': 'AOI updated successfully'}), 200
        except KeyError as e:
            print(f"Missing key in AOI data: {e}")
            return jsonify({'error': 'Missing keys in AOI data'}), 400
    else:
        return jsonify({'error': 'Invalid data provided'}), 400

 
#function to run Flask app
def run_flask_app():
    app.run(debug=False, host='0.0.0.0', port=5000)

#main entry point for the script 
if __name__ == '__main__':
    # Start the Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.start()

    # Add any code here that needs to run in parallel to the Flask app

    # Wait for the Flask thread to complete
    flask_thread.join()
