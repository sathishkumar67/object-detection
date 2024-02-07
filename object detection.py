from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import cv2
import numpy as np
import pyttsx3

# Load Yolos model and image processor
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

# Set up text-to-speech engine
engine = pyttsx3.init()
engine.say("XPoSat (X-ray Polarimeter Satellite) is India’s first dedicated polarimetry mission to study various dynamics of bright astronomical X-ray sources in extreme conditions. The spacecraft will carry two scientific payloads in a low earth orbit. The primary payload POLIX (Polarimeter Instrument in X-rays) will measure the polarimetry parameters (degree and angle of polarization) in medium X-ray energy range of 8-30 keV photons of astronomical origin. The XSPECT (X-ray Spectroscopy and Timing) payload will give spectroscopic information in the energy range of 0.8-15 keV.The emission mechanism from various astronomical sources such as blackhole, neutron stars, active galactic nuclei, pulsar wind nebulae etc. originates from complex physical processes and are challenging to understand. While the spectroscopic and timing information by various space based observatories provide a wealth of information, the exact nature of the emission from such sources still poses deeper challenges to astronomers. The polarimetry measurements add two more dimension to our understanding, the degree of polarization and the angle of polarization and thus is an excellent diagnostic tool to understand the emission processes from astronomical sources. The polarimetric observations along with spectroscopic measurements are expected to break the degeneracy of various theoretical models of astronomical emission processes. This would be the major direction of research from XPoSat by Indian science community.")
engine.runAndWait()





# # OpenCV setup for webcam
# cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera, change it if you have multiple cameras
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Convert OpenCV BGR image to RGB (PIL format)
#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#     # Model inference
#     inputs = image_processor(images=image, return_tensors="pt")
#     outputs = model(**inputs)
#
#     # Post-process and get results
#     target_sizes = torch.tensor([image.size[::-1]])
#     results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
#
#     # Variable to check if a person is detected
#     person_detected = False
#
#     # Visualize the results on the frame
#     for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#         box = [round(i, 2) for i in box.tolist()]
#
#         # Draw bounding box on the frame
#         cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
#
#         # Display class label and confidence
#         label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
#         cv2.putText(frame, label_text, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#         # Check if a person is detected
#         if model.config.id2label[label.item()] == "person":
#             person_detected = True
#
#     # If a person is detected, say "There is a person"
#     if person_detected:
#         engine.say("There is a person")
#         engine.runAndWait()
#
#     # Display the resulting frame
#     cv2.imshow('Object Detection', frame)
#
#     # Increase the frame speed by reducing the delay (e.g., 1 millisecond)
#     if cv2.waitKey(3) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
