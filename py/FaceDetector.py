# OpenCV documentation: https://docs.opencv.org/2.4/index.html
import cv2
from random import randrange

# load some pre-trained  data of front-facing faces using OpenCV (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('../xml/haarcascade_frontalface_default.xml')


# Detect faces from images
# choose an image to detect the face (imread: image read, an OpenCV furnction)
img = cv2.imread('../img/picture1.png')

# convert image to grayscole
# cvtColor(variable_of_the_image, cv2.BGR2GRAY: BlueGreenRedToGray). In OpenCV, it is BGR instead of RGB.
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

# detect faces and store the coordinates, width and height of the rectangle that contain the face.
# only the top right coordinate of the rectangle will be given. With the width and the height, the rectangle can be drawn.
# [[x y width height], [x, y, w, h], ... ] this is a list in another list
#detectMultiScale(): no matter the how small the face is, detectMultiScale() will still be able to detect the faces in the image. It is only looking for the overall composition of the face - relation between the eyes to the nose and to the mouth
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img) 

# the for-loop will loop over the tuple 
for (x, y, w, h) in face_coordinates:
    # draw rectangles around the faces
    # (img_in_color, (x, y), (x + width, y + height) , (Blue,Green,Red), thickness_of_rectangle)
    # randrange: provides any color between 0 and 256
    print(cv2.rectangle(img, (x, y), (x+w,y+h), (randrange(256), randrange(256), randrange(256)), 2)) 
    
# display the image (imshow: image show, an OpenCV function) 
# imshow('name of window that display the image', variable_of_the_image)
cv2.imshow('Covid19-Mask-Detector', img) 

# pause the execution until any key is pressed
cv2.waitKey()

print("It is working!")

"""
# Detect faces from a video

# capture video from webcam
# 0 is the default value and it reads from the webcam
# cv2.VideoCapture('img/video1.mp4') will read from the video with name video1.mp4 
webcam = cv2.VideoCapture(0) 

while True:
    # read current frame from video and returns 2 things: a boolean and the actual frame being read
    successful_frame_read, frame = webcam.read()

    # convert the frame to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces and store the coordinates, width and height of the rectangle that contain the face.
    # only the top right coordinate of the rectangle will be given. With the width and the height, the rectangle can be drawn.
    # [[x y width height], [x, y, w, h], ... ] this is a list in another list
    #detectMultiScale(): no matter the how small the face is, detectMultiScale() will still be able to detect the faces in the image. It is only looking for the overall composition of the face - relation between the eyes to the nose and to the mouth
    face_coordinates = trained_face_data.detectMultiScale(grayscale_frame) 

    # the for-loop will loop over the tuple 
    for (x, y, w, h) in face_coordinates:
        # draw rectangles around the faces
        # (img_in_color, (x, y), (x + width, y + height) , (Blue,Green,Red), thickness_of_rectangle)
        # randrange: provides any color between 0 and 256
        print(cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 255, 0), 2)) 

    cv2.imshow('Covid19-Mask-Detector', frame)
    key = cv2.waitKey(1)
"""