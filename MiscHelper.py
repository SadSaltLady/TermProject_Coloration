import numpy as np
from scipy import interpolate

def coordToIdx(i, j, dim):
    return i * dim[1] + j


def addGradPadding(chnl):
  '''
  add padding to the top and left side of chnl, 
  returns array of shape (chnl.x +1, chnl.y + 1)'''
  #currently they just pad as zero since it won't affect my final result
  w, h = np.shape(chnl)
  padded = np.zeros((w + 1, h + 1))
  padded[1:, 1:] = chnl
  return padded
#taken from assignment 3
def addDivPadding(chnl):
  '''
  add padding to the bot and right side of chnl, 
  returns array of shape (chnl.x +1, chnl.y + 1)'''
  #currently they just pad as zero since it won't affect my final result
  w, h = np.shape(chnl)
  padded = np.zeros((w + 1, h + 1))
  padded[0:-1, 0:-1] = chnl
  return padded

def grad(chnl):
  '''calculate the gradient using np.diff
  returns a vector field of size (shape(chnl), 2)'''

  padded = addGradPadding(chnl)
  partialy = np.diff(padded, axis=0)[:, 1:]
  partialx = np.diff(padded, axis=1)[1: ]

  return np.stack((partialx, partialy), axis= -1)



