import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import inv

import src.helpers.plot as P


def findChessboardCorners(glob_in, cols = 9, rows = 6, factor = 1, plot = True, dir_out = False):
    
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images
    obj_points = [] # 3d points in real world space
    img_points = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(glob_in)

    # Get a grid to plot the images
    gs = P.getGridFor(len(images), 2)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img = cv2.resize(img, (0,0), fx = factor, fy = factor) 
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        # If found, add object points, image points
        if ret == True:
            obj_points.append(objp)
            img_points.append(corners)

            # Draw and display the corners
            if plot or dir_out:
                cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
                name = 'corners_found_' + str(idx)
            
                if plot:
                    ax = plt.subplot(gs[int(idx / 2), idx % 2])
                    ax.imshow(img[:,:,::-1]) # BRG TO RGB
                    ax.set_title(name)
                    
                if dir_out:
                    cv2.imwrite(dir_out + name + '.jpg', img)
        elif plot:
            name = 'corners_not_sfound_' + str(idx)
            ax = plt.subplot(gs[int(idx / 2), idx % 2])
            ax.imshow(img[:,:,::-1]) # BRG TO RGB
            ax.set_title(name)
            
    return obj_points, img_points


def calibrate_camera(obj_points, img_points, size, output):
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    # Save the camera calibration result for later use:
    data = {}
    data["mtx"] = mtx
    data["dist"] = dist
    data["rvecs"] = rvecs
    data["tvecs"] = tvecs
    
    pickle.dump(data, open(output, "wb"))
    
    return mtx, dist


def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def getM(src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    
    return M, inv(M)
    
    
def warper(img, M, size):
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
