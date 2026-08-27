import cv2 as cv

fd=cv.CascadeClassifier(r'../samples/Cascades/haarcascade_frontalface_default.xml')
vidcap= cv.VideoCapture(0)


while True:
    ret , frame = vidcap.read()
    frame = cv.resize(frame,(980, 720) )
    frame = cv.flip(frame, 1)
    imgrey=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    det=fd.detectMultiScale(imgrey, minSize=(220,220))

    for (x,y,w,h) in det:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
        print(w,h)

    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break