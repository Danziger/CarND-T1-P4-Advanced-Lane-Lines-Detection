from collections import deque
from sklearn.linear_model import LinearRegression

import cv2
import numpy as np


class Line:

    def __init__(self, size=8):
        self.size = size
        self.reset()

    def reset(self):
        # was the line detected in the last iteration?
        self.detected = False  
        
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        
        # radius of curvature of the line in some units
        self.radius_of_curvature = None 
        
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        
        # x values for detected line pixels
        self.allx = None
        
        # y values for detected line pixels
        self.ally = None
        
        # ORIGINAL:

        size = self.size

        self.coefsBuffer = deque(maxlen=size)
        self.weights = list(range(1, size+1, 1))
        
        self.coefs = []
        self.xs = []
        self.ys = []
        self.fitx = []

        
    def push(self, coefs, xs, ys, fitx, ploty):
        self.detected = True
        self.coefsBuffer.append(coefs)
        
        currentItems = len(self.coefsBuffer)
        
        coefs = np.average(self.coefsBuffer, axis=0, weights=self.weights[-currentItems:self.size])
        xs = np.polyval(coefs, ploty)
        ys = ploty
        fitx = xs
        
        self.coefs = coefs
        self.xs = xs
        self.ys = ys
        self.fitx = fitx
        
        # TODO: Check similarity
        
        # TODO: Average coefs and update others...
        
        return coefs, xs, ys, fitx
    
    
    def getLast(self):
        self.detected = False
        
        currentItems = len(self.coefsBuffer)
        
        coefs = np.average(self.coefsBuffer, axis=0, weights=self.weights[-currentItems:self.size])
        xs = np.polyval(coefs, self.ys)
        fitx = xs
        
        self.coefsBuffer.append(coefs)
        self.coefs = coefs
        self.xs = xs
        self.fitx = fitx
        
        return coefs, xs, self.ys, fitx
    
