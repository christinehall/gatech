# Final Project - Computational Photography
# Christine Hall, chall61@gatech.edu
# ARTSTIGATOR

import numpy as np
import scipy.signal
import cv2
import scipy.sparse
import math
import pyamg
import sys

def run():
    
    print 'Argument List:', str(sys.argv)

    artwork_string = str(sys.argv[1]) + ".jpg"

    try:
        artwork = cv2.imread(artwork_string)
        fask = cv2.imread("fask2.jpg", cv2.IMREAD_GRAYSCALE)
        if artwork.shape[:2] != (500,500):
            # artwork = smart_resize(artwork)
            print "Image is not in correct size"
            return
    except:
        print "One or more of your photos can't be read right now..."

    num_faces = int(sys.argv[2])
    artwork_gray = cv2.cvtColor(artwork, cv2.COLOR_BGR2GRAY)
    l, w = artwork.shape[:2]

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
    mouth_cascade = cv2.CascadeClassifier('mouth.xml')
    # nose_cascade = cv2.CascadeClassifier('nose.xml')


    # detect the faces in the artwork and the selfie
    original_faces = face_cascade.detectMultiScale(artwork_gray, 1.3, 5)


    if len(original_faces) <= 1:
        print "There are not enough faces to swap here!!"
    else:
        print "Found {0} selfie faces!".format(len(original_faces))
    
    mask = np.zeros(artwork.shape[:2])
    swap = np.zeros(artwork.shape)
    all_faces = []

    for i in range(0, num_faces):
        this_face = Face(np.zeros((original_faces[i][2], original_faces[i][3],3)))
        this_face.x = original_faces[i][0]
        this_face.y = original_faces[i][1]
        this_face.w = original_faces[i][2]
        this_face.h = original_faces[i][3]

        # cv2.rectangle(artwork, (this_face.x, this_face.y), (this_face.x + this_face.w, this_face.y + this_face.h), (255,0,0), 10)
        for x in range(this_face.x, this_face.x + this_face.w):
            for y in range(this_face.y, this_face.y + this_face.h):
                this_face.img[y - original_faces[i][1], x - original_faces[i][0]] = artwork[y,x]
        
        image_name = "face_"+ str(i) + ".jpg"
        cv2.imwrite(image_name, this_face.img)

        this_face.gray_img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        this_face.eyes = eye_cascade.detectMultiScale(this_face.gray_img, 1.2, 5)
        print "Number of eyes found in this face: " + str(len(this_face.eyes))

        if len(this_face.eyes) == 2:
            this_face.angle = -1 * calculate_angle(this_face.eyes[0], this_face.eyes[1])
            print this_face.eyes
            print "angle for face " + str(i) + ": " + str(this_face.angle)
        else:
            this_face.angle = 0

        # this_face.nose = nose_cascade.detectMultiScale(this_face.gray_img, 1.2, 4)
        # cv2.imwrite("nose" + str(i) + ".jpg", this_face.nose)

        # this_face.mouth = mouth_cascade.detectMultiScale(this_face.gray_img, 1.3, 5)
        # cv2.imwrite("mouth" + str(i) + ".jpg", this_face.nose)
    
        all_faces.append(this_face)

    # cv2.imwrite("lines.jpg", artwork)

    for index in range(0, num_faces):
        if index != num_faces - 1:
            swap_index = index + 1
        else:
            swap_index = 0

        origin_face = all_faces[index]
        swap_face = all_faces[swap_index]

        origin_face_resized = cv2.resize(origin_face.img, swap_face.img.shape[:2])
        difference_in_angle = swap_face.angle - origin_face.angle
        origin_face_resized = rotate_image(origin_face_resized, -1 * origin_face.angle)
        origin_face_resized = rotate_image(origin_face_resized, swap_face.angle)
        fask_resized = cv2.resize(fask, swap_face.img.shape[:2])
        fask_resized = rotate_image(fask_resized, swap_face.angle)
        cv2.imwrite("thefask.jpg", fask_resized)

        # write it to the swap image
        startx = swap_face.x
        starty = swap_face.y
        width = swap_face.w
        length = swap_face.h
        center = ((width + startx - startx) / 2, (length + starty - starty) / 2)
        # mask = cv2.circle(mask, center, width/2, 255)#, 255, -1)
        # cv2.imwrite("eekmask.jpg", mask)


        for x in range(startx, startx+width):
            for y in range(starty, starty+length):
                swap[y,x] = origin_face_resized[y-starty, x-startx]
                if mask[y,x] != 255:
                    mask[y,x] = fask_resized[y-starty, x-startx]


    cv2.imwrite("mask_test.jpg", mask)
    cv2.imwrite("swapimage.jpg", swap)

    final_image = poisson_blend(artwork, swap, mask)

    image_name = str(sys.argv[1]) + "_swapped.jpg"
    cv2.imwrite(image_name, final_image)

def smart_resize(img):
    print "Resizing the image"
    w,h = img.shape[:2]
    if w < 500 and h < 500:
        # what
        pass
    else:
        print "Image is > 500,500"
        x_crop = w - 500
        y_crop = h - 500
        crop_img = img[y_crop/2:500, x_crop/x:500]
    return crop_img

def calculate_angle(eye_1, eye_2):
    # eye_1 and eye_2 will be x,y,w,

    deltaY = eye_1[1] * 1.0 - eye_2[1] * 1.0
    deltaX = eye_1[0] * 1.0 - eye_2[0] * 1.0

    angleInDegrees = math.atan2(deltaY, deltaX) * 180.0 / math.pi

    if angleInDegrees > 90:
        angleInDegrees = 180 - angleInDegrees
    elif angleInDegrees < -90:
        angleInDegrees = 180 + angleInDegrees

    return angleInDegrees

def rotate_image(image, angle):
    center=tuple(np.array(image.shape[0:2])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[0:2],flags=cv2.INTER_LINEAR)

def poisson_blend(artwork_img, selfie_img, mask_img):
    # this function works only for 500x500 images (input adjustment needeed)
    # kudos to https://github.com/fbessho/PyPoi
    mask_img = mask_img[0:500, 0:500]
    mask_img[mask_img == 0] = False
    mask_img[mask_img != False] = True

    # create coefficient matrix
    A = scipy.sparse.identity(np.prod((500,500)), format='lil')
    for y in range(0,500):
        for x in range(0,500):
            if mask_img[y, x]:
                index = x + y * 500
                A[index, index] = 4
                if index + 1 < np.prod((500,500)):
                    A[index, index + 1] = -1
                if index - 1 >= 0:
                    A[index, index - 1] = -1
                if index + 500 < np.prod((500,500)):
                    A[index, index + 500] = -1
                if index - 500 >= 0:
                    A[index, index - 500] = -1

    A = A.tocsr() # Returns a copy of this matrix in Compressed Sparse Row format

    # create poisson matrix for b
    P = pyamg.gallery.poisson(mask_img.shape)

    # for each layer (ex. RGB)
    for channel in range(artwork_img.shape[2]):
        t = artwork_img[0:500, 0:500, channel]
        s = selfie_img[0:500, 0:500, channel]
        t = t.flatten() # Returns a copy of the array collapsed into one dimension.
        s = s.flatten()

        # create b
        b = P * s
        for y in range(0,500):
            for x in range(0,500):
                if not mask_img[y, x]:
                    index = x + y * 500
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(A, b, verb=False, tol=1e-10)

        # assign x to target image
        x = np.reshape(x, (500,500))
        x[x > 255] = 255
        x[x < 0] = 0
        x = np.array(x, artwork_img.dtype)
        artwork_img[0:500, 0:500, channel] = x

    return artwork_img

class Face:

    def __init__(self, face_img):
        self.img = face_img
        self.angle = None
        self.size = None
        self.gray_img = None
        self.x = None
        self.y = None
        self.w = None
        self.h = None
        self.eyes = None
        self.nose = None
        self.mouth = None

if __name__ == "__main__":
    print "Starting FaceSwapper! Stay tuned while we swap your faces"
    run()