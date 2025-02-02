import numpy as np
import cv2
import matplotlib.pyplot as plt

import src.helpers.cameraCalibration as C


def getFirstTime(binary_warped, hist_slice=2, nwindows = 12, minpix = 60, margin = 100):
    # Choose the number of sliding windows
    # Set the width of the windows +/- margin
    # Set minimum number of pixels found to recenter window

    height = binary_warped.shape[0]
    width = binary_warped.shape[1]
    center = width // 2
        
    # Set height of windows
    window_height = height // nwindows
    hist_slice = height // hist_slice if hist_slice else window_height
    
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[-hist_slice:,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    leftx_base = np.argmax(histogram[:center])
    rightx_base = np.argmax(histogram[center:]) + center
    
    # TODO: In sharp turns we cannot assume this. We need to find peaks independently of their position, maybe just relaying on previous
    # curvature. That could also be used to create wider/narrower and dynamic margins!

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # TODO: Maybe fit a line each time to decide how much we should move the boxes in the harder video?
            
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img, leftx, rightx, lefty, righty


def getFromRegion(binary_warped, coefs_L, coefs_R, margin = 100):
    # TODO: Separate ploting from calculations...
    
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (coefs_L[0]*(nonzeroy**2) + coefs_L[1]*nonzeroy + 
        coefs_L[2] - margin)) & (nonzerox < (coefs_L[0]*(nonzeroy**2) + 
        coefs_L[1]*nonzeroy + coefs_L[2] + margin))) 

    right_lane_inds = ((nonzerox > (coefs_R[0]*(nonzeroy**2) + coefs_R[1]*nonzeroy + 
        coefs_R[2] - margin)) & (nonzerox < (coefs_R[0]*(nonzeroy**2) + 
        coefs_R[1]*nonzeroy + coefs_R[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    coefs_L = np.polyfit(lefty, leftx, 2)
    coefs_R = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = coefs_L[0]*ploty**2 + coefs_L[1]*ploty + coefs_L[2]
    right_fitx = coefs_R[0]*ploty**2 + coefs_R[1]*ploty + coefs_R[2]
    
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    return coefs_L, coefs_R, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img, ploty, left_fitx, right_fitx, leftx, rightx, lefty, righty


# TODO: Probably, it's better to do it in a different way for the harder video: An insightful student has suggested an
# alternative approach which may scale more efficiently. That is, once the parabola coefficients are obtained, in pixels,
# convert them into meters. For example, if the parabola is x= a*(y**2) +b*y+c; and mx and my are the scale for the x
# and y axis, respectively (in meters/pixel); then the scaled parabola is x= mx / (my ** 2) *a*(y**2)+(mx/my)*b*y+c

def getRadius(ploty, leftx, rightx, lefty, righty):
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/500 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad


def getDistanceFromCenter(coefs_L, coefs_R, screen_center, y = 0, m_per_px_x = 1):
    lx_at_y = np.polyval(coefs_L, y)
    rx_at_y = np.polyval(coefs_R, y)
    
    lane_center = (lx_at_y + rx_at_y) / 2
    
    deviation = screen_center - lane_center
    
    # Return value in px and m
    return deviation, deviation * m_per_px_x


def drawOverlay(warped, undist, Minv, left_fitx, right_fitx, ploty, size):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,0), thickness=16)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(255,0,0), thickness=16)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = C.warper(color_warp, Minv, size)
    
    # Combine the result with the original image
    return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


def drawOverlayLane(warped, undist, Minv, left_fitx, right_fitx, ploty, size, radius, deviation):
    img = drawOverlay(warped, undist, Minv, left_fitx, right_fitx, ploty, size)
    
    return drawOverlayInfo(img, radius, deviation)


def drawOverlayInfo(img, radius, deviation):
    side = 'OK'
    
    if deviation <= 0.005:
        side = 'LEFT'
    elif deviation >= 0.005:
        side = 'RIGHT'
    
    text = 'LANE RADIUS: %.2f KM  DEVIATION FROM CENTER: %.2f M %s' % (radius/1000, abs(deviation), side)
    
    cv2.rectangle(img, (0, 0), (1080, 85), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, text, (30, 55), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

    return img
