import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import amax
from scipy import io, misc, sparse, signal, ndimage
from scipy.sparse import coo
from scipy.sparse.linalg import spsolve
import cv2 
import os 

from MiscHelper import *


def generateDifferenceMask(img1, img2):
    diff = cv2.absdiff(img1, img2)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)[1]

    #erode it a bit to get rid of noise
    kernel = np.ones((2,2), np.uint8)
    diff = cv2.erode(diff, kernel, iterations=1)
    return diff

def getNeighbors(i, j, s):
    neighbors = []
    #hopefully if statements are faster than forloops
    h, w = s[:2]
    if i > 0:
        neighbors.append((i-1, j))
    if i < h-1:
        neighbors.append((i+1, j))
    if j > 0:
        neighbors.append((i, j-1))
    if j < w-1:
        neighbors.append((i, j+1))
    if i > 0 and j > 0:
        neighbors.append((i-1, j-1))
    if i < h-1 and j < w-1:
        neighbors.append((i+1, j+1))
    if i > 0 and j < w-1:
        neighbors.append((i-1, j+1))
    if i < h-1 and j > 0:
        neighbors.append((i+1, j-1))
    return neighbors

def weightCalculation(neighbors, loc, Y):
    #calculate variance at point
    values = []
    weights = []
    self_val = Y[loc[0], loc[1]]
    for n in neighbors:
        values.append(Y[n[0], n[1]])
    al = values.copy()
    al.append(self_val)
    all_vals = np.array(al)
    var = np.var(np.array(all_vals))

    bottom = var * 2.0
    if (bottom < 0.00001):
        return np.ones(len(neighbors))
    else:
        #calculate the weights of each neighbor
        weights = np.exp(- np.square(self_val - values) / bottom)
        return weights

def weightCalculationV2(neighbors, loc, Y):
    #calculate variance at point
    values = []
    weights = []
    self_val = Y[loc[0], loc[1]]
    for n in neighbors:
        values.append(1.0 / ((abs(Y[n[0], n[1]]) + 1) ** 6))

    return values


def pixelDiffernent(gray, cues, i, j):
    diff = gray[i, j] - cues[i, j]
    if sum(abs(diff)) > 0.001:
        return True
    else:
        return False


#test their code and see if it works
def solver(gray, cues, mask, size, saliency):
    n, m = cues.shape[0], cues.shape[1]
    W = sparse.lil_matrix((size, size), dtype = float)
    bu = np.zeros(shape = (size))
    bv = np.zeros(shape = (size))
    Y_gray = gray[::,::, 0]
    for i in range(n):
        for j in range(m):
            #if this pixel is marked
            if pixelDiffernent(gray, cues, i, j):
                id = coordToIdx(i, j, gray.shape)
                W[id, id] = 1.0
                bu[id] = cues[i, j, 1]
                bv[id] = cues[i, j, 2]
            else:
                #else set up the problem base on how differnt pixels are from their neighbors
                id = coordToIdx(i, j, gray.shape)
                neighbour = getNeighbors(i, j, gray.shape)
                #calculate the weights according to the equation
                weights = weightCalculationV2(neighbour, (i, j), saliency) 
                sum_weights = sum(weights)
                weights = weights/sum_weights #normalize
                for k in range(len(neighbour)):
                    id_y = coordToIdx(neighbour[k][0], neighbour[k][1], gray.shape)
                    W[id, id_y] += -1 * weights[k] 
                W[id, id] += 1.
    W = W.tocsc()
    u = spsolve(W, bu)
    v = spsolve(W, bv)

    print(np.amax(u), np.amax(v))
    print(np.amin(u), np.amin(v))
    
    return u, v

def getGradientOrientation(grad):
    return np.arctan2(grad[::, ::, 1], grad[::, ::, 0])

def getGradientMagnitude(grad):
    return np.sqrt(np.square(grad[::, ::, 0]) + np.square(grad[::, ::, 1]))

def getGradientNormalized(mag, fsize):
    #find average in a window by blurring
    blur = ndimage.uniform_filter(mag, size = fsize)
    #calculate gradient based on this
    #still needs a big for loop sadly
    var_all = np.zeros(mag.shape)
    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            lower_x = max(0, i - fsize)
            upper_x = min(mag.shape[0], i + fsize)
            lower_y = max(0, j - fsize)
            upper_y = min(mag.shape[1], j + fsize)
            size = (upper_x - lower_x) * (upper_y - lower_y)
            window = mag[lower_x:upper_x, lower_y:upper_y]
            variance = np.sum(np.square(window - blur[i, j]))/(size**2)
            var_all[i, j] = np.sqrt(variance)
            #normalize
    eps = 0.00001
    ret = (mag - blur) / (var_all + eps)
    return ret

def getBilinearWeights(x, y):
    #calculate the weights of each neighbor
    lx = np.floor(x)
    rx = np.ceil(x)
    ly = np.floor(y)
    ry = np.ceil(y)

    weights = [(rx - x)*(ry - y), (x - lx)*(ry - y), (rx - x)*(y - ly), (x - lx)*(y - ly)]
    weights = np.array(weights) / ((rx - lx) * (ry - ly) + 0.00001)
    return weights

def gerOrientationWeights(orient):
    #convole with the four neighbors
    conv_filter = np.array([[0, -1, 0], [-1, 1, -1], [0, -1, 0]])
    weights = ndimage.convolve(orient, conv_filter, mode = 'constant')
    weights = np.exp(- np.square(weights) / (2 * np.pi / 5))
    return weights 

def edgeLines(pos_x, pos_y, orient, m):
    #find bilinear interpolation weights
    weights4D = getBilinearWeights(pos_x, pos_y)
    #find the weights for the orientation
    orientWeights = gerOrientationWeights(orient)
    #multiply the weights with the gradient
    weight = weights4D[0] * orientWeights[0] + weights4D[1] * orientWeights[1] + weights4D[2] * orientWeights[2] + weights4D[3] * orientWeights[3]
    return weight/4.0

def edgelengthCalculation(orient, normalize,iter):
    m0 = np.zeros(orient.shape)
    m1 = np.zeros(orient.shape)
    sqrttwo = np.sqrt(2)

    while (iter > 0):
        pos_y = sqrttwo * np.cos(orient)
        pos_x = sqrttwo * np.sin(orient)
        neg_y = sqrttwo * np.cos(orient + np.pi)
        neg_x = sqrttwo * np.sin(orient + np.pi)
        
        #positive direction
        weight_pos = edgeLines(pos_x, pos_y, orient, m0)
        m0 = weight_pos * (m0 + normalize)
        #negative direction
        weight_neg = edgeLines(neg_x, neg_y, orient, m1)
        m1 = weight_neg * (m1 + normalize)

        iter -= 1
    
    #find edgelength from everthing calculated
    edge_len = m0 + m1 + normalize
    return edge_len



def saliencyCalculation(gray, fsize):
    #find matrix of 
    print(np.shape(gray))

    gradient = grad(gray) #note: this is of (x, y) format
    gradientVisualize = np.sqrt(np.square(gradient[::, ::, 0]) + np.square(gradient[::, ::, 1]))
    gradOrient = getGradientOrientation(gradient)
    gradMagnitude = getGradientMagnitude(gradient)
    #gradOrient = gradOrient * 180 / np.pi #convert to degrees
    

    gradNP = getGradientNormalized(gradMagnitude, fsize)
    NPnormalized = (gradNP + np.amin(gradNP))/(np.amax(gradNP) - np.amin(gradNP))


    edgelen_est = edgelengthCalculation(gradOrient, gradNP, 60)
    #FINALL, saliency 
    sx = np.square(np.cos(gradOrient)) * edgelen_est * gradient[::, ::, 0]
    sy = np.square(np.sin(gradOrient)) * edgelen_est * gradient[::, ::, 1]

    saliency_visualization = np.sqrt(np.square(sx) + np.square(sy))

    return saliency_visualization


def main():
    baseimage = cv2.imread('data/ex2.png')
    markimage = cv2.imread('data/ex2_marked.png')
    
    print("Image shape: ", baseimage.shape)
    assert(baseimage.shape == markimage.shape)
    

    h, w = baseimage.shape[:2]


    diff = baseimage - markimage
    base = cv2.cvtColor(baseimage, cv2.COLOR_RGB2YUV) /255.0
    mark = cv2.cvtColor(markimage, cv2.COLOR_RGB2YUV) /255.0

    Y = base[:,:,0]

    
    #try to do saliency calculation
    saliency = saliencyCalculation(Y, 5)
    saliency = grad(Y)[::, ::, 0]
    Y = base[:,:,0]
    u, v = solver(base, mark, diff, h*w, saliency)

    #reshape the solved U, V
    u = u.reshape((h, w))
    v = v.reshape((h, w))
    result = np.stack((Y, u, v), axis = 2)
    #cv doesn't like it unless I do this
    result = (np.clip(result, 0., 1.) * 255).astype(np.uint8)

    result = cv2.cvtColor(result, cv2.COLOR_YUV2RGB)


    #show result 
    cv2.imwrite('data/result_ex2.png', result)
    cv2.imshow('result', result)
    cv2.waitKey(0)

    #weight = generateWeightMatrix(base, mark, diff, h*w)




main()