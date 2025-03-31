import cv2

face_detect = cv2.CascadeClassifier("/home/glitch00/data/Projects/Computer Vision/Face Detection/haarcascade_frontalface_default.xml")
eye_detect = cv2.CascadeClassifier("/home/glitch00/data/Projects/Computer Vision/Face Detection/haarcascade_eye.xml")

stream = cv2.VideoCapture(0)
while True : 
    st,frame = stream.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    ##### Face Detect

    faces = face_detect.detectMultiScale(gray,1.3,2)
    for (x,y,w,h) in faces :
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        face =  frame[y:y+h,x:x+w]
        #### eye Detect
        eyes = eye_detect.detectMultiScale(face)
        for (ex,ey,ew,eh) in eyes : 
            cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)


    ##### Show
    cv2.imshow("Live Stream",frame)
    cv2.waitKey(33)
