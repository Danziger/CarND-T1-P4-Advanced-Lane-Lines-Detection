import numpy as np

import src.helpers.imageProcessing as IP
import src.helpers.cameraCalibration as C


def pipeline(img, M):
    WIDTH = img.shape[1]
    HEIGHT = img.shape[0]
    
    imgBW = IP.rgb2gray(img)
    imgHLS = IP.rgb2hls(img)
    
    imgHLS_S = imgHLS[:,:,2]
    
    hls_yw_filter = IP.yellowAndWhiteHlsFilter(imgHLS)
    
    laplacian = IP.lap_threshold(imgBW, kernel=31)
    laplacianS = IP.lap_threshold(imgHLS_S, kernel=31, thresh=0.075)

    laplacianBlur = IP.gaussianBlur(laplacian, kernel_size=7)
    laplacianSBlur = IP.gaussianBlur(laplacianS, kernel_size=7)
    
    filtered = np.zeros_like(hls_yw_filter)
    filtered[(hls_yw_filter == 1) | (laplacianSBlur >= 0.75) | (laplacianBlur >= 0.75)] = 1

    REGION_MASK = np.array([[   
        (0, 400),
        (WIDTH, 400),
        (WIDTH-250, 700),
        (WIDTH/2, 650),
        (250, 700),
    ]], dtype=np.int32)

    roi = IP.regionOfInterestFilter(filtered, REGION_MASK)
    
    binary = C.warper(roi, M, (WIDTH, HEIGHT))

    return binary
    