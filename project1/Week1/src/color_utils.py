import cv2
import numpy as np

def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def video_rgb2gray(video):
    return np.array([rgb2gray(frame) for frame in video])
