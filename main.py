import cv2

'''
利用圓周移動來平移邊界框（物體的位置）。
簡單地說，KCF追踪器關注圖像中的變化方向（可能是運動、延伸或方向），
並試圖生成要追踪的物體的概率位置。
'''

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (540,960))

tracker = cv2.TrackerKCF_create()
video = cv2.VideoCapture('mimi.mp4')

ret,frame=video.read()
bbox = cv2.selectROI(frame)

# Initialize tracker with first frame and bounding box
ret = tracker.init(frame,bbox)

while True:
    ret,frame=video.read()
    if not ret:
        break

    ret,bbox=tracker.update(frame)

    if ret:
        (x,y,w,h)=[int(v) for v in bbox]
        cv2.rectangle(frame,(x,y+h-20),(x+w+20,y+h+20),(0,255,0),-1)
        cv2.putText(frame,'MiMi',(x+20,y+h+10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.rectangle(frame,(x,y),(x+w+20,y+h+20),(0,255,0),2,2)
    else:
        cv2.putText(frame,'Error',(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow('Tracking',frame)
    out.write(frame)
    if cv2.waitKey(1) & 0XFF==27:
        break

cv2.destroyAllWindows()

