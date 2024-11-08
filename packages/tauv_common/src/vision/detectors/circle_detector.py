import cv2 as cv
import os
import skimage.morphology as morph
import numpy as np
KERNEL_SIZE = 5
MIN_AREA = 200
MIN_POINTS = 5

def filter(img):
    #color filter images
    hsl = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    lower1 = np.array([0,20,30])
    upper1 = np.array([120,100,180])
    res = cv.inRange(hsl, lower1, upper1)

    #perform some morphological sharpening
    kernel1 = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    kernel3 = np.ones((7,7),np.uint8)

    res = cv.erode(res,kernel1,iterations = 2)
    res = cv.dilate(res,kernel2,iterations = 1)
    res = cv.morphologyEx(res, cv.MORPH_CLOSE, kernel3)
    res = cv.morphologyEx(res, cv.MORPH_OPEN, kernel1)
    res = cv.ximgproc.thinning(res)

    return res

def ellipseArea(ellipse):
    (x, y), (MA, ma), angle = ellipse
    return (np.pi*MA*ma/4.0)


def findEllipse(contours, num=2):
    arDif = np.zeros(contours.shape[0])
    ellipses = np.empty(contours.shape[0], dtype=np.object0)
    #ellipses = []
    counter = 0
    for contour in contours:
        hull = cv.convexHull(contour)
        if(len(hull)<MIN_POINTS):
            continue
        
        ellipse = cv.fitEllipse(hull)

        fitArea=ellipseArea(ellipse)
        contourArea = cv.contourArea(hull)

        if(contourArea<MIN_AREA):
            continue

        arDif[counter] = np.abs(fitArea - contourArea)

        ellipses[counter] = ellipse
        counter+=1
    
    points = np.argsort(arDif)
    return ellipses[points[0:min(len(points),num)]]


def findShapes(img, num=4):
    #find contours
    contours, h =cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    if(len(contours)<1):
        return ([],[])

    #filter contours by size
    areas = [cv.arcLength(contour, closed=False) for contour in contours]
    maxAreas = np.argsort(areas)[max(0, len(areas)-num):]
    res = np.take(contours, maxAreas)

    return findEllipse(res)#, findPoly(res))


def detectCircles(frameImg):
    res = filter(frameImg)

    ellipses = findShapes(res)

    return ellipses