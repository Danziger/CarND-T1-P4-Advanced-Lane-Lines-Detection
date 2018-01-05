import cv2
import numpy as np


# COLOR SPACES:

def rgb2hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def rgb2hls(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


def rgb2lab(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2Lab)


def hls2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_HLS2GRAY)


def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def bgr2gray(img):
    # Use when reading an image with cv2.imread()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# IMAGE PROCESSING:

def eq(img): # TODO: Rename to incrementContrast
    img = img.copy()
    
    data32 = np.asarray(img, dtype="int32")
    data32[:, :] = (data32[:, :] - 50) * 1.125

    np.clip(data32, 0, 255, out=data32)

    return  data32.astype('uint8')


# HLS FILTERS:

def yellowHlsFilter(img, lower=[15, 25, 155], upper=[45, 200, 255]):
    # H in [0, 180], S in [0, 255], L in [0, 255]
    # See http://hslpicker.com

    lower = np.uint8(lower)
    upper = np.uint8(upper)

    return cv2.inRange(img, lower, upper)


def whiteHlsFilter(img, lower=[0, 200, 155], upper=[180, 255, 255]):
    # H in [0, 180], S in [0, 255], L in [0, 255]
    # See http://hslpicker.com

    lower = np.uint8(lower)
    upper = np.uint8(upper)

    return cv2.inRange(img, lower, upper)


def yellowAndWhiteHlsFilter(img):
    return cv2.bitwise_or(yellowHlsFilter(img), whiteHlsFilter(img))


# HLS-H FILTERS:

def hlsHFilter(img, thresh=(15, 50)):
    binary = np.zeros_like(img)
    binary[(img > thresh[0]) & (img <= thresh[1])] = 1
    
    return binary


# HLS-S FILTERS:

def hlsSFilter(img, thresh=(150, 255)):
    binary = np.zeros_like(img)
    binary[(img > thresh[0]) & (img <= thresh[1])] = 1
    
    return binary


# Lab FILTERS:

def labYellowFilter(img, lower=[100, 110, 150], upper=[255, 160, 255]):
    # L in [0, 255], a in [0, 255], b in [0, 255]
    # See http://hslpicker.com

    lower = np.uint8(lower)
    upper = np.uint8(upper)

    return cv2.inRange(img, lower, upper)


# Lab-b FITLERS:

def labBFilter(img, thresh=(150, 255)):
    binary = np.zeros_like(img)
    binary[(img > thresh[0]) & (img <= thresh[1])] = 1
    
    return binary
    

# RGB IMAGE + FILTER STACKED:

def yellowAndWhiteRgbFiltered(img):
    return cv2.bitwise_and(img, img, mask=yellowAndWhiteHlsFilter(rgb2hls(img)))


# REGION OF INTEREST / MASKING:

def regionOfInterestFilter(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    return cv2.bitwise_and(img, mask)


# SOBEL

def abs_sobel_thresh(gray, orient='x', thresh_min=20, thresh_max=255):
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1 if orient is 'x' else 0, 1 if orient is 'y' else 0)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    
    mask = np.zeros_like(scaled_sobel)
    mask[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return mask


def mag_thresh(gray, sobel_kernel=9, mag_thresh=(55, 80)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    scaled = np.uint8(255 * magnitude / np.max(magnitude))
    
    mask = np.zeros_like(scaled)
    mask[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1

    return mask


def dir_threshold(gray, sobel_kernel=31, thresh=(0.7, 1.3)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    direction = np.arctan2(abs_sobely, abs_sobelx)
    
    mask = np.zeros_like(direction)
    mask[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    
    return mask

def lap_threshold(gray, kernel=31, thresh=0.125):
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=kernel)
    
    # Values are too small to set the threshold manually, so better set it based on the smallest value
    
    thresh = thresh * np.min(laplacian)
    
    mask = np.zeros_like(laplacian)
    mask[laplacian < thresh] = 1
    
    return mask

def gaussianBlur(img, kernel_size=3):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def houghLines(img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)


def drawLines(img, lines, color=(255, 0, 0), thickness=10, make_copy=True):
    # Copy the passed image
    img_copy = np.copy(img) if make_copy else img

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)

    return img_copy


def classifyLinesPoints(lines, width):
    centerLeft = width * 0.4
    centerRight = width * 0.6

    # Classify the points based on their slope:

    rightPointsX = []
    rightPointsY = []
    rightWeights = []
    rightLines = []
    leftPointsX = []
    leftPointsY = []
    leftWeights = []
    leftLines = []
    discardedLines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)

            # Ignore suspicious lines:

            slopeAbs = np.abs(slope)
            slopeMin = 1 / 3
            slopeMax = 3

            if slopeAbs < slopeMin or slopeAbs > slopeMax:
                discardedLines.append([[x1, y1, x2, y2]])

                continue

            # If line is valid, classify it in right or left based on its slope, and keep its length as well:

            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            if slope > 0 and x1 > centerLeft and x2 > centerLeft:
                rightPointsX.extend([x1, x2])
                rightPointsY.extend([y1, y2])
                rightWeights.extend([length, length])
                rightLines.append([[x1, y1, x2, y2]])
            elif slope < 0 and x1 <= centerRight and x2 <= centerRight:
                leftPointsX.extend([x1, x2])
                leftPointsY.extend([y1, y2])
                leftWeights.extend([length, length])
                leftLines.append([[x1, y1, x2, y2]])
            else:
                discardedLines.append([[x1, y1, x2, y2]])

    return dict(
        right=dict(X=rightPointsX, Y=rightPointsY, weights=rightWeights, lines=rightLines),
        left=dict(X=leftPointsX, Y=leftPointsY, weights=leftWeights, lines=leftLines),
        discarded=discardedLines,
    )


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)
