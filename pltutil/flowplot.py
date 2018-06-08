'''
Created on 19 Dec 2014

@author: jadarve
'''

import pkg_resources
import numpy as np
import scipy.ndimage.interpolation as interp
import scipy.misc as misc


__all__ = ['flowToColor']

# load the optical flow color wheel texture
__colorWheelTexture = misc.imread(pkg_resources.resource_filename('pltutil.rsc', 'colorWheelTexture.png'), flatten=False)

__colorWheel_R = np.copy(__colorWheelTexture[:,:,0])
__colorWheel_G = np.copy(__colorWheelTexture[:,:,1])
__colorWheel_B = np.copy(__colorWheelTexture[:,:,2])



def flowToColor(flowField, maxFlow=1.0):
    
    global __colorWheel_R
    global __colorWheel_G
    global __colorWheel_B
    
    h = __colorWheel_R.shape[0]
    w = __colorWheel_R.shape[1]
    
    OF_m = np.zeros(flowField.shape)
    OF_m[:,:,:] = (flowField + maxFlow) / float(2*maxFlow)
    OF_m[:,:,0] *= (w-1)
    OF_m[:,:,1] *= (h-1)
    OF_m = np.reshape(OF_m, (flowField.shape[0]*flowField.shape[1], 2)).T
    
    OF_flip = np.zeros_like(OF_m)
    OF_flip[0,:] = OF_m[1,:]
    OF_flip[1,:] = OF_m[0,:]
    
    color_R = np.zeros((flowField.shape[0]*flowField.shape[1]), dtype=np.uint8)
    color_G = np.zeros_like(color_R)
    color_B = np.zeros_like(color_R)
    
    interp.map_coordinates(__colorWheel_R, OF_flip, color_R, order=0, mode='nearest', cval=0)
    interp.map_coordinates(__colorWheel_G, OF_flip, color_G, order=0, mode='nearest', cval=0)
    interp.map_coordinates(__colorWheel_B, OF_flip, color_B, order=0, mode='nearest', cval=0)
    
    flowColor = np.zeros((flowField.shape[0], flowField.shape[1], 3), dtype=np.uint8)
    flowColor[:,:,0] = color_R.reshape(flowField.shape[0:2])
    flowColor[:,:,1] = color_G.reshape(flowField.shape[0:2])
    flowColor[:,:,2] = color_B.reshape(flowField.shape[0:2])
    
    return flowColor

def colorWheel():
    return __colorWheelTexture
