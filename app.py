import cv2
import time
import numpy as np

def getFaceBox(net, frame, conf_threshold=0.75):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

    return frameOpencvDnn, bboxes

# Load the models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-4)', '(5-10)', '(11-15)', '(15-20)', '(20-30)', '(30-40)', '(40-50)', '(60-100)']
genderList = ['Male', 'Female']

ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

cap = cv2.VideoCapture(0)
padding = 20

# Processing interval in seconds
processing_interval = 10
last_processed_time = time.time()

while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break

    current_time = time.time()

    # Create a smaller frame for better optimization
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frameFace, bboxes = getFaceBox(faceNet, small_frame)

    if not bboxes:
        print("No face detected, checking next frame")
        continue

    if current_time - last_processed_time >= processing_interval:
        age_preds = np.zeros((1, len(ageList)))
        gender_preds = np.zeros((1, len(genderList)))

        for bbox in bboxes:
            face = small_frame[max(0, bbox[1]-padding):min(bbox[3]+padding, small_frame.shape[0]-1),
                               max(0, bbox[0]-padding):min(bbox[2]+padding, small_frame.shape[1]-1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Predict gender
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender_preds += genderPreds
            
            # Predict age
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age_preds += agePreds

        gender = genderList[gender_preds.argmax()]
        age = ageList[age_preds.argmax()]

        print("Gender: {}, confidence: {:.3f}".format(gender, gender_preds.max()))
        print("Age: {}, confidence: {:.3f}".format(age, age_preds.max()))

        label = "{},{}".format(gender, age)
        for bbox in bboxes:
            cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        last_processed_time = current_time

    cv2.imshow("Age Gender Demo", frameFace)