import numpy as np

import src.helpers.laneFinder as LF


def validateLines(coefs_L, coefs_R, HEIGHT):
    # Checking that they have similar curvature
    # Checking that they are separated by approximately the right distance horizontally
    # Checking that they are roughly parallel

    LANE_WIDTH = 500
    MARGIN = 100
    MAX_DIFF = 75
    
    #times = np.abs(coefs_L / coefs_R)
    
    #if times[0] < 0.125 or times[0] > 8:
    #    print('y2 too different')
        
    #    return False

    #if times[1] < 0.25 or times[1] > 4:
    #    print('y too different')
        
    #    return False 
    
    Y = np.linspace(0, HEIGHT - 1, 8)
    lane_width = np.polyval(coefs_R, Y) - np.polyval(coefs_L, Y)
    
    if max(lane_width) - min(lane_width) > MAX_DIFF:
        print('MAX_DIFF not ok')
        
        return False
    
    if np.any(lane_width < LANE_WIDTH - MARGIN) or np.any(lane_width > LANE_WIDTH + MARGIN):
        print('width not ok')
        
        return False
    
    return True
   
# TODO: Maybe check width in right and left and similarity in current and previous...
    
def pipeline(img, binary, Minv, line_L, line_R):
    WIDTH = img.shape[1]
    HEIGHT = img.shape[0]
    CENTER = WIDTH // 2
    
    # LINE DETECTION & VALIDATION:
    
    linesOk = False
    
    if line_L.detected and line_R.detected:
        # Search inside margins:

        coefs_L, coefs_R, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img, \
            ploty, left_fitx, right_fitx, leftx, rightx, lefty, righty = LF.getFromRegion(binary, line_L.coefs, line_R.coefs)
            
        linesOk = validateLines(coefs_L, coefs_R, HEIGHT)
    
    if not linesOk:
        # If lines were not detected or the detected ones are not valid, search again from scratch:
        
        coefs_L, coefs_R, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img, \
            leftx, rightx, lefty, righty = LF.getFirstTime(binary, hist_slice=5, nwindows = 12, minpix = 60, margin = 100)

        # Generate x and y values for plotting
        # TODO: ploty is always the same, this could be reused...
        ploty = np.linspace(0, HEIGHT - 1, HEIGHT)
        left_fitx = np.polyval(coefs_L, ploty)
        right_fitx = np.polyval(coefs_R, ploty)
        
        linesOk = validateLines(coefs_L, coefs_R, HEIGHT)

        
    # LINE SMOTHERING:

    if linesOk:
        # If both lines are ok, try to add them to the Line instances:
        coefs_L, leftx, lefty, left_fitx = line_L.push(coefs_L, leftx, lefty, left_fitx, ploty)
        coefs_R, rightx, righty, right_fitx = line_R.push(coefs_R, rightx, righty, right_fitx, ploty)
        
    else:
        # Otherwise, use the previous one:
        coefs_L, leftx, lefty, left_fitx = line_L.getLast()
        coefs_R, rightx, righty, right_fitx = line_R.getLast()
        
        # TODO: Could we try to predict it instead?
        
    if len(coefs_L) == 0 or len(coefs_R) == 0:
        print("Skip")
        
        return img
        
    # TODO: Could individual ok lines be used?
    # TODO: Could a single line be used and simulate the other one?
    # TODO: Could both lines' curvature be averaged together?
        
    # Overlay:
    
    # TODO: Add a "confident" indicator based on how we got the N previous lines?
    # TODO: Add a "noise" indicator?
    # TODO: Add other indicators?
    
    rad_m_L, rad_m_R = LF.getRadius(ploty, leftx, rightx, lefty, righty)
    deviation_px, deviation_m = LF.getDistanceFromCenter(coefs_L, coefs_R, CENTER, HEIGHT, 3.7/500)
    
    return LF.drawOverlayLane(binary, img, \
        Minv, left_fitx, right_fitx, ploty, (WIDTH, HEIGHT), (rad_m_L + rad_m_R) / 2, deviation_m)