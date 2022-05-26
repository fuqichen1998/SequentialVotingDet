import torch
import numpy as np

# 100doh dataloader helper functions
def threshold(x, low, high):
    return min(max(x, low), high)

def enlarge_box(box, h_thres, w_thres, factor):
    x1, y1, x2, y2 = box
    dh = (y2-y1)*(factor-1)//2
    dw = (x2-x1)*(factor-1)//2
    x1 = threshold(x1-dw, 0, w_thres)
    y1 = threshold(y1-dh, 0, h_thres)
    x2 = threshold(x2+dw, 0, w_thres)
    y2 = threshold(y2+dh, 0, h_thres)
    return [x1, y1, x2, y2]

def cropped_resized_x(x, s, c, r, low, high):
    """
    return the corresponding index in cropped then resized image to
    the original image
    
    Inputs:
        x: original coordinates ratio
        s: original size
        c: crop idx
        r: resized ratio
    Output:
        x': corresponding index in cropped then resized image
    """
    return threshold((x*s-c)*r, low, high)