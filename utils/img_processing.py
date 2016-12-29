import dlib
import cv2
import random
import math
import numpy as np
from skimage import io
from face_dlib import AlignDlib
from scipy.spatial.distance import euclidean

def scale_img(img, shortest=None, largest=None):
    if shortest is not None:
        shortest_edge = np.argmin(img.shape)
        if shortest_edge == 0:
            ratio = float(shortest) / img.shape[0]
            new_width = img.shape[1] * ratio
            img = cv2.resize(img, (int(new_width), shortest))
        else:
            ratio = float(shortest) / img.shape[1]
            new_height = img.shape[0] * ratio
            img = cv2.resize(img, (shortest, int(new_height)))

    if largest is not None:
        largest_edge = np.argmax(img.shape)
        if largest_edge == 0:
            ratio = float(largest) / img.shape[0]
            new_width = img.shape[1] * ratio
            img = cv2.resize(img, (int(new_width), largest))
        else:
            ratio = float(largest) / img.shape[1]
            new_height = img.shape[0] * ratio
            img = cv2.resize(img, (largest, int(new_height)))

    if largest is None and shortest is None:
        print "No resizing!"
    return img

def crop_face(img, rect):
    x1 = int(rect.left())
    x2 = int(rect.right())
    y2 = int(rect.bottom())
    y1 = int(rect.top())

    pad_x1, pad_x2, pad_y1, pad_y2 = 0, 0, 0, 0

    if x1 < 0:
        pad_x1 = np.abs(x1)

    if x2 > img.shape[1]:
        pad_x2 = x2 - img.shape[1]

    if y1 < 0:
        pad_y1 = np.abs(y1)

    if y2 < img.shape[0]:
        pad_y2 = np.abs(y2 - img.shape[0])

    if img.ndim == 3:
        padding = [(pad_y1, pad_y2),(pad_x1,pad_x2), (0,0)]

    if img.ndim == 2:
        padding = [(pad_y1,pad_y2),(pad_x1, pad_x2)]

    x1 = x1 if x1 > 0  else 0
    y1 = y1 if y1 > 0  else 0

    img = np.pad(img, padding, mode='constant', constant_values=114.4)
    if img.ndim == 3:
        face_img = img[y1:y2, x1:x2, :]
    elif img.ndim == 2:
        face_img = img[y1:y2, x1:x2]
    return face_img

def pad_img(img, target_size):
    w_p1 = int(np.floor((target_size - img.shape[1]) /2.0))
    w_p2 = int(np.ceil((target_size - img.shape[1]) /2.0))
    h_p1 = int(np.floor((target_size - img.shape[0]) /2.0))
    h_p2 = int(np.ceil((target_size - img.shape[0]) /2.0))

    if w_p1 < 0:
        w_p1 = 0
    if w_p2 < 0:
        w_p2 = 0
    if h_p1 < 0:
        h_p1 = 0
    if h_p2 < 0:
        h_p2 = 0

    if img.ndim == 3:
        padding = ((h_p1, h_p2),(w_p1, w_p2),(0,0))
    elif img.ndim == 2:
        padding = ((h_p1, h_p2),(w_p1, w_p2))

    return np.pad(img, padding,
         mode = 'constant', constant_values = 127.0)

def process_img(img, fdet, size=None, is_shortest=True, pad=False, margins=None, bbox=None):
    """
    TODO: Do not warp face image, use computed rotation matrix to rotate
    face bounding box. It enables to rotate compute both upper body image and
    face image together.

    Process image:
    1. Face detection if not bbox given
    2. Landmark detection for face alignment and centering
    3. Crop face image with upper body margins
    4. Center and rotate face image
    5. Pad empty regions with constant 117
    """
    if bbox is None:
        print "ERROR: No given bbox for the image !!"
        return False
    else:
        rect = dlib.rectangle(left=int(np.round(bbox['x1'])),
                              top=int(np.round(bbox['y1'])),
                              right=int(np.round(bbox['x2'])),
                              bottom=int(np.round(bbox['y2'])))

    # if rect is None and bbox is not None:
    if rect is not None:
        lms = fdet.findLandmarks(img,rect)

    # find generic bb locations - solving different bboxes coming from different devices
    f_x1, f_y1 = 999999999, 999999999
    f_x2, f_y2 = -1, -1
    for lm in lms:
        if lm[0] < f_x1:
            f_x1 = lm[0]

        if lm[1] < f_y1:
            f_y1 = lm[1]

        if lm[1] > f_y2:
            f_y2 = lm[1]

        if lm[0] > f_x2:
            f_x2 = lm[0]

    # crop face with new bb
    rect = dlib.rectangle(left=int(np.round(f_x1)),
                          top=int(np.round(f_y1)),
                          right=int(np.round(f_x2)),
                          bottom=int(np.round(f_y2)))

    face_img = crop_face(img, rect)

    # take eyes
    le_x = lms[37][0]
    le_y = lms[37][1]

    re_x = lms[44][0]
    re_y = lms[44][1]
    center = (rect.center().x, rect.center().y)

    # Normal bb dimensions
    bb_height = rect.height()
    bb_width = rect.width()

    # Margins
    if margins is None:
        width_margin = bb_width *0.5
        bot_margin = bb_height  *0.75
        top_margin = bb_height  *0.75
    else:
        width_margin = bb_width * margins['width']
        bot_margin = bb_height  * margins['bottom']
        top_margin = bb_height  * margins['top']

    # Crop Coordinateswith given margin relative to face bbox center
    top_coor = int(center[1]- bb_height/2 - top_margin)
    top_coor = top_coor if top_coor >= 0 else 0

    bot_coor = int(center[1] + bb_height/2 + bot_margin)
    bot_coor = bot_coor if bot_coor <= img.shape[0] else img.shape[0]

    left_coor = int(center[0] - bb_width/2 - width_margin)
    left_coor = left_coor if left_coor >= 0 else 0

    right_coor = int(center[0] + bb_width/2 + width_margin)
    right_coor = right_coor if right_coor <= img.shape[1] else img.shape[1]

    # find rotation angle
    angle =  np.arctan2(le_y - re_y, re_x - le_x) * 180 / np.pi
    M = cv2.getRotationMatrix2D(center, -angle, 1)

    # Cropped upright rectangle
    cropped = cv2.warpAffine(img, M,
                             (img.shape[1],img.shape[0]),
                              borderValue = [127.0, 127.0, 127.0])
    if cropped.ndim == 3:
        final_img = cropped[top_coor:bot_coor, left_coor:right_coor, :]
    elif cropped.ndim == 2:
        final_img = cropped[top_coor:bot_coor, left_coor:right_coor]

    if size is not None:
        if is_shortest:
            final_img = scale_img(final_img, shortest = size)
        else:
            final_img = scale_img(final_img, largest = size)

    if pad == True:
        final_img = pad_img(final_img, size)

    return final_img, face_img
