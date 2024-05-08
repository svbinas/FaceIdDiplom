import cv2
import numpy as np
import face_recognition
import os
from PIL import ImageGrab
import time

path = 'images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')
fps = 3.0
#cap = cv2.VideoCapture(0)
cv2.namedWindow('Screen', cv2.WINDOW_NORMAL)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'MP4V' for .mp4
out = cv2.VideoWriter('output.mp4', fourcc, fps, (1920, 1080))
prev_frame_time = time.time()

while True:
    screen = np.array(ImageGrab.grab(bbox=None)) # bbox specifies specific region (bbox=left, top, right, bottom)
    img = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    out.write(img)
    imgS = cv2.resize(img, (0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    # Write the frame to the video file

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img,name, (x1+6, y2-6), cv2.FONT_HERSHEY_PLAIN,1, (255, 255, 255), 2)

    cv2.imshow('Screen', img)
    # Wait to maintain the desired frame rate
    time_elapsed = time.time() - prev_frame_time
    delay = max(1.0 / fps - time_elapsed, 0)
    time.sleep(delay)
    prev_frame_time = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break
out.release()
cv2.destroyAllWindows()
