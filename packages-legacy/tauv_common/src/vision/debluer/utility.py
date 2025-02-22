import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from vision.debluer.PSO import *

# shows histogram of all 3 channels 
def color_hist(img):

    y = np.linspace(0 ,256)
    fig , ax = plt.subplots(3,1)
    ax[0].hist(img[:,:,0].flatten().ravel(),color='blue',bins = 256)
    ax[1].hist(img[:,:,1].flatten().ravel(),color='green',bins = 256)
    ax[2].hist(img[:,:,2].flatten().ravel(),color='red',bins = 256)

    plt.show()

def plot_hist(img):

    plt.hist(img.flatten(),bins = 150)
    plt.show()

# stacking BGR channels in order after computation
def image(input):
    val = list(input)

    for p in range(len(val)):
        if val[p][1]=="B":
            b = val[p][0]
        elif val[p][1]=="G":
            g = val[p][0]
        if val[p][1]=="R":
            r = val[p][0]
    img = np.dstack([b,g,r])
    img = np.array(img,dtype=np.uint8)

    return img

# Indicating superior, inferior and intermediate channels based on mean of pixels in channel
def superior_inferior_split(img):

    B, G, R = cv.split(img)

    pixel = {"B":np.mean(B) ,"G":np.mean(G),"R":np.mean(R)}
    pixel_ordered = dict(sorted(pixel.items(), key=lambda x: x[1], reverse=True))

    # Classifying Maximum, Minimum and Intermediate channels of image
    label =["Pmax","Pint","Pmin"]
    chanel={}

    for i,j in zip(range(len(label)),pixel_ordered.keys()):
         if j=="B":
             chanel[label[i]]=list([B,j])
        
         elif j=="G":
             chanel[label[i]]=list([G,j])
            
         else:
             chanel[label[i]]=list([R,j])

    return chanel


def neutralize_image(img):

    track = superior_inferior_split(img)

    Pmax = track["Pmax"][0]
    Pint = track["Pint"][0]
    Pmin = track["Pmin"][0]

    #gain_factor Pint
    J = (np.sum(Pmax) - np.sum(Pint))/(np.sum(Pmax) + np.sum(Pint))

    #gain_factor Pmin
    K = (np.sum(Pmax) - np.sum(Pmin))/(np.sum(Pmax) + np.sum(Pmin))

    track["Pint"][0] = Pint + (J * Pmax)
    track["Pmin"][0] = Pmin + (K * Pmax)

    #neutralised image
    neu_img = image(track.values())

    return neu_img
def Stretching(image):

    LSR_img = [] # for lower stretched image
    USR_img = [] # for upper stretched image
    height, width = image.shape[:2]

    for i in range(image.shape[2]):
        img_hist = image[:,:,i]
        max_P = np.max(img_hist)
        min_P = np.min(img_hist)

        mean_P = np.mean(img_hist)
        median_P = np.median(img_hist)

        avg_point = (mean_P + median_P)/2

        LS_img = np.zeros((height, width))
        US_img = np.zeros((height, width))

        mask = np.zeros((height, width))
        for i in range(0, height):
            for j in range(0, width):
                if img_hist[i][j] < avg_point:
                    LS_img[i][j] = int((( img_hist[i][j] - min_P) * ((255 - min_P) / (avg_point - min_P)) + min_P))
                    print(f'{((255 - min_P) / (avg_point - min_P))}')
                    US_img[i][j] = 0
                    mask[i][j] = 255
                    #array_upper_histogram_stretching[i][j] = p_out
                else:
                    LS_img[i][j] = 255
                    US_img[i][j] = int((( img_hist[i][j] - avg_point) * ((255) / (max_P - avg_point))))

        # cv2.imshow(f'msk {i}', mask)
        LSR_img.append(LS_img)
        USR_img.append(US_img)

    LS = np.array(np.dstack(LSR_img),dtype=np.uint8)
    US = np.array(np.dstack(USR_img),dtype=np.uint8)

    cv2.imshow('ls', LS)
    cv2.imshow('us', US)
    cv2.waitKey(1)

    return LS,US

def Stretching_new(image):
    LSR_img = np.zeros_like(image, dtype=np.uint8)
    USR_img = np.zeros_like(image, dtype=np.uint8) # for upper stretched image
    height, width = image.shape[:2]

    ones = np.full((height,width), 255, dtype=np.uint8)

    for i in range(image.shape[2]):
        img_hist = image[:,:,i]
        max_P = np.max(img_hist)
        min_P = np.min(img_hist)

        mean_P = np.mean(img_hist)
        median_P = np.median(img_hist)

        avg_point = (mean_P + median_P)/2

        mask = (img_hist < avg_point).astype(np.uint8)
        inv_mask = cv2.bitwise_not(mask)

        # print(f'{inv_mask=}')
        LS_img_pre = (((img_hist - min_P) * ((255 - min_P) / (avg_point - min_P)) +
                       min_P)).astype(np.uint8)
        if i == 1:
            cv2.imshow("pre", LS_img_pre)
        LSR_img[:, :, i] = cv2.bitwise_or(LS_img_pre, LS_img_pre, mask=mask)
        LSR_img[:, :, i] = cv2.bitwise_or(LSR_img[:,:,i], ones, mask=inv_mask)

        US_k = (255 / (max_P - avg_point))
        US_img_pre = (img_hist - avg_point) * US_k
        USR_img[:, :, i] = cv2.bitwise_or(US_img_pre, US_img_pre, inv_mask)

    cv2.imshow('ls', LSR_img)
    cv2.imshow('us', USR_img)
    cv2.waitKey(1)
    return LSR_img, USR_img

    #
    #     #
    #     # for i in range(0, height):
    #     #     for j in range(0, width):
    #     #         if img_hist[i][j] < avg_point:
    #     #             LS_img[i][j] = int((( img_hist[i][j] - min_P) * ((255 - min_P) / (avg_point - min_P)) + min_P))
    #     #             US_img[i][j] = 0
    #     #             #array_upper_histogram_stretching[i][j] = p_out
    #     #         else:
    #     #             LS_img[i][j] = 255
    #     #             US_img[i][j] = int((( img_hist[i][j] - avg_point) * ((255) / (max_P - avg_point))))
    #     #
    #     # LSR_img.append(LS_img)
    #     # USR_img.append(US_img)
    #
    # # LS = np.array(np.dstack(LSR_img),dtype=np.uint8)
    # # US = np.array(np.dstack(USR_img),dtype=np.uint8)
    # LSR_img = LSR_img.astype(np.uint8)



def enhanced_image(img1, img2):

    #integration of dual intensity images to get Enhanced-constrast output image
    b1,g1,r1 = cv.split(img1)
    b2,g2,r2 = cv.split(img2)

    height, width = img1.shape[:2]
    dual_img=np.zeros((height, width,3),dtype=np.uint8)

    dual_img[:,:,0] = np.array(np.add(b1/2, b2/2),dtype = np.uint8)
    dual_img[:,:,1] = np.array(np.add(g1/2, g2/2),dtype = np.uint8)
    dual_img[:,:,2] = np.array(np.add(r1/2, r2/2),dtype = np.uint8)

    return dual_img


def pso_image(img):

    group = superior_inferior_split(img)

    maxi = np.mean(group["Pmax"][0])
    inte = np.mean(group["Pint"][0])
    mini = np.mean(group["Pmin"][0])

    # Defining hyperparameters
    n = 50  # number of particles
    params = {"wmax" : 0.9, "wmin" : 0.4, "c1" : 2 , "c2" : 2}
    max_iteration = 10

    x = np.array([inte, mini])

    def func(X,P_sup = maxi):
        return np.square(P_sup - X[0])+np.square(P_sup - X[1])

    nVar= 2  # number of variables to optimize
    VarMin = 0  # lower bound of variables , you can use np.array() for different variables
    VarMax = 255   # upper bound of variables, you can use np.array() for different variables

    gbest = pso(func, max_iter=max_iteration, num_particles = n, dim = 2, vmin = VarMin, vmax = VarMax, params = params)

    #gamma correction for inferior color channels
    mean_colors = gbest['position']
    gamma = np.log(mean_colors/255)/np.log(x/255)

    group["Pint"][0] = np.array(255*np.power(group["Pint"][0]/255 , gamma[0]))
    group["Pmin"][0] = np.array(255*np.power(group["Pmin"][0]/255 , gamma[1]))


    pso_res = image(group.values())

    return pso_res

def unsharp_masking(img):

    alpha = 0.2
    beta = 1 -alpha
    img_blur = cv.GaussianBlur(img, (1,1),sigmaX=1)
    unsharp_img = cv.addWeighted(img, alpha, img_blur, beta, 0.0)

    return unsharp_img

def NUCE(img):
    print('neutralizing...')
    #superior based underwater color cast neutralization
    neu_img = neutralize_image(img)
    print('stretch')
    #Dual-intensity images fusion based on average of mean and median values
    img1, img2 = Stretching_new(neu_img)

    print('enhance')
    dual_img = enhanced_image(img1, img2)

    print('pso')
    #Swarm-intelligence based mean equalization
    pso_res = pso_image(dual_img)

    print('unsharp')
    #Unsharp masking
    nuce_img = unsharp_masking(pso_res)

    print('done')
    print()
    return nuce_img
