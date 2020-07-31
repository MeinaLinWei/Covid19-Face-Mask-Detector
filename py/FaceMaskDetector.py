import cv2
from random import randrange

import cv2
from random import randrange

# load some pre-trained data of front-facing faces using OpenCV (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('../xml/haarcascade_frontalface_default.xml')

# load some pre-trained data of mouths using OpenCV
trained_mouth_data = cv2.CascadeClassifier('../xml/haarcascade_mcs_mouth.xml')

# load some pre-trained data of noses using OpenCV
trained_nose_data = cv2.CascadeClassifier('../xml/haarcascade_mcs_nose.xml')

# load some pre-trained data ofeyes using OpenCV
trained_eye_data = cv2.CascadeClassifier('../xml/haarcascade_eye.xml')


# Detect faces from images
# choose an image to detect the face (imread: image read, an OpenCV furnction)
img = cv2.imread('../img/picture6.jpg')

# convert image to grayscole
# cvtColor(variable_of_the_image, cv2.BGR2GRAY: BlueGreenRedToGray). In OpenCV, it is BGR instead of RGB.
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

# detect faces and store the coordinates, width and height of the rectangle that contain the face.
# only the top right coordinate of the rectangle will be given. With the width and the height, the rectangle can be drawn.
# [[x y width height], [x, y, w, h], ... ] this is a list in another list
#detectMultiScale(): no matter the how small the face is, detectMultiScale() will still be able to detect the faces in the image. It is only looking for the overall composition of the face - relation between the eyes to the nose and to the mouth
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img) 
eye_coordinates = trained_eye_data.detectMultiScale(grayscaled_img) 
nose_coordinates = trained_nose_data.detectMultiScale(grayscaled_img) 
mouth_coordinates = trained_mouth_data.detectMultiScale(grayscaled_img) 


eyes = len(eye_coordinates)
nose = len(nose_coordinates)
mouth = len(mouth_coordinates)
# the for-loop will loop over the tuple 
for (x, y, w, h) in face_coordinates:
    # draw rectangles around the faces
    # (img_in_color, (x, y), (x + width, y + height) , (Blue,Green,Red), thickness_of_rectangle)
    # randrange: provides any color between 0 and 256
    print(cv2.rectangle(img, (x, y), (x+w,y+h), (randrange(256), randrange(256), randrange(256)), 2))


if (eyes != 0):
        if ((nose != 0) and (mouth != 0)):
            # the for-loop will loop over the tuple 
            for (x, y, w, h) in face_coordinates:
                # draw rectangles around the faces
                # (img_in_color, (x, y), (x + width, y + height) , (Blue,Green,Red), thickness_of_rectangle)
                # randrange: provides any color between 0 and 256
                print(cv2.rectangle(img, (x, y), (x+w,y+h), (0, 0, 255), 2)) 
                print(cv2.putText(img, 'No mask detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2))

        if ((nose == 0 ) and (mouth == 0)):
            for (x, y, w, h) in face_coordinates:
                print(cv2.rectangle(img, (x, y), (x+w,y+h), (0, 255, 0), 2)) 
                print(cv2.putText(img, 'Mask detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 0), 2 ))
            
        if ((nose != 0) and (mouth ==0)):
            for (x, y, w, h) in face_coordinates:
                print(cv2.rectangle(img, (x, y), (x+w,y+h), (0, 0, 255), 2)) 
                print(cv2.putText(img, 'Please cover your nose also', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.80 , (0, 0, 255), 2))

        if ((nose == 0) and (mouth != 0)):
            for (x, y, w, h) in face_coordinates:
                print(cv2.rectangle(img, (x, y), (x+w,y+h), (0, 0, 255), 2)) 
                print(cv2.putText(img, 'Please cover your mouth also', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.80 , (0, 0, 255), 2)) 
    
# display the image (imshow: image show, an OpenCV function) 
# imshow('name of window that display the image', variable_of_the_image)
cv2.imshow('Covid19-Mask-Detector', img) 

# pause the execution until any key is pressed
cv2.waitKey()


"""
# Detect faces from a video

# capture video from webcam
# 0 is the default value and it reads from the webcam
# cv2.VideoCapture('img/video1.mp4') will read from the video with name video1.mp4 
vid = cv2.VideoCapture(0) 

while True:
    # read current frame from video and returns 2 things: a boolean and the actual frame being read
    successful_frame_read, frame = vid.read()

    # convert the frame to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces and store the coordinates, width and height of the rectangle that contain the face.
    # only the top right coordinate of the rectangle will be given. With the width and the height, the rectangle can be drawn.
    # [[x y width height], [x, y, w, h], ... ] this is a list in another list
    #detectMultiScale(): no matter the how small the face is, detectMultiScale() will still be able to detect the faces in the image. It is only looking for the overall composition of the face - relation between the eyes to the nose and to the mouth
    face_coordinates = trained_face_data.detectMultiScale(grayscale_frame) 
    mouth_coordinates = trained_mouth_data.detectMultiScale(grayscale_frame, 1.7, 11)
    nose_coordinates = trained_nose_data.detectMultiScale(grayscale_frame, 1.3, 5)
    eye_coordinates = trained_eye_data.detectMultiScale(grayscale_frame)

    eyes = len(eye_coordinates)
    nose = len(nose_coordinates)
    mouth = len(mouth_coordinates)


    if (eyes != 0):
        if ((nose != 0) and (mouth != 0)):
            # the for-loop will loop over the tuple 
            for (x, y, w, h) in face_coordinates:
                # draw rectangles around the faces
                # (img_in_color, (x, y), (x + width, y + height) , (Blue,Green,Red), thickness_of_rectangle)
                # randrange: provides any color between 0 and 256
                print(cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 0, 255), 2)) 
                print(cv2.putText(frame, 'No mask detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2))


        if ((nose == 0 ) and (mouth == 0)):
            for (x, y, w, h) in face_coordinates:
                print(cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 255, 0), 2)) 
                print(cv2.putText(frame, 'Mask detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 0), 2 ))
            
        if ((nose != 0) and (mouth ==0)):
            for (x, y, w, h) in face_coordinates:
                print(cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 0, 255), 2)) 
                print(cv2.putText(frame, 'Please cover your nose also', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.80 , (0, 0, 255), 2))

        if ((nose == 0) and (mouth != 0)):
            for (x, y, w, h) in face_coordinates:
                print(cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 0, 255), 2)) 
                print(cv2.putText(frame, 'Please cover your mouth also', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.80 , (0, 0, 255), 2))


    
    


    cv2.imshow('Covid19-Mask-Detector', frame)
    key = cv2.waitKey(1)

"""