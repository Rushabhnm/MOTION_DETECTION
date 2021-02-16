import cv2, time, pandas
from datetime import datetime

first_frame = None

status_list = [None,None]                                             # to get the details of incoming & outgoing time of the object
time_recorded =[]                                                     # all the time stamp would be appended in the list
data_frame = pandas.DataFrame(columns=["Start","End"])

video = cv2.VideoCapture(0)                                           # capturing live video through primary camera (i.e 0)

while True:
    check, frame = video.read()

    status = 0                                                        # status initially set to 0 for the 1st frame

    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame,(21,21),0)               # 21,21-width and the height (parameters of the the blurness)
                                                                      # 0 - standard deviation


    if first_frame is None:
        first_frame = gray_frame
        continue


    delta_frame = cv2.absdiff(first_frame,gray_frame)
# absolute difference between the first frame taken with the next frames(i.e the background and then the preceding frames)


    threshold_frame = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
# setting a threshold level and highlighting those ares with white color(255) which are above the threshold


    threshold_frame = cv2.dilate(threshold_frame,None,iterations=2)
# for smoothening out the white color which will represent the objects in motion


    (cnts,_)= cv2.findContours(threshold_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# finding the contours in the threshold frame

    for contours in cnts:
        if cv2.contourArea(contours)<1000:
            continue
# if the area of the contour found is less than 10000 pixels ignore that contour.

        status = 1                                                          # updating the status to 1 as there is an object detected
        print("OBJECTED DETECTED!!")
        (x,y,w,h)=cv2.boundingRect(contours)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

    status_list.append(status)


    if status_list[-1]==1 and status_list[-2]==0:
        time_recorded.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        time_recorded.append(datetime.now())


    cv2.imshow("Gray Frame", gray_frame)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Threshold Frame",threshold_frame)
    cv2.imshow("Color Frame",frame)
# displaying all the windows


    key=cv2.waitKey(1)

    if key == ord('q'):
        if status == 1:
            time_recorded.append(datetime.now())
        break

print(status_list)
print(time_recorded)

for i in range(0,len(time_recorded),2):
    data_frame = data_frame.append({"Start":time_recorded[i],"End":time_recorded[i+1]},ignore_index=True)

data_frame.to_csv("TIMES.csv")
video.release()
