import numpy as np

import src.helpers.laneFinder as LF


def pipeline(img, binary, Minv):
    WIDTH = img.shape[1]
    HEIGHT = img.shape[0]
    CENTER = WIDTH // 2
    
    # Fit:
    coefs_L, coefs_R, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img, \
        leftx, rightx, lefty, righty = LF.getFirstTime(binary, hist_slice=5, nwindows = 12, minpix = 60, margin = 100)
        
    # Generate x and y values for plotting
    ploty = np.linspace(0, HEIGHT - 1, HEIGHT)
    left_fitx = np.polyval(coefs_L, ploty)
    right_fitx = np.polyval(coefs_R, ploty)
    
    # Overlay:
    
    rad_m_L, rad_m_R = LF.getRadius(ploty, leftx, rightx, lefty, righty)
    deviation_px, deviation_m = LF.getDistanceFromCenter(coefs_L, coefs_R, CENTER, HEIGHT, 3.7/500)
    
    return LF.drawOverlayLane(binary, img, \
        Minv, left_fitx, right_fitx, ploty, (WIDTH, HEIGHT), (rad_m_L + rad_m_R) / 2, deviation_m)