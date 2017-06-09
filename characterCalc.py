import helpers
import cv2
import numpy as np
import math
import data

def calcAll(points, frame):
    characters = []
    return calcAngle(points) + calcEyes(points) + calcEyebrow(points, frame) + calcMouth(points)

def calcAngle(shape):
    image_points = np.array([
        (shape[30][0], shape[30][1]),  # Nose tip
        (shape[8][0], shape[8][1]),  # Chin
        (shape[5][0], shape[5][1]),
        (shape[11][0], shape[11][1]),
        (shape[36][0], shape[36][1]),  # Left eye left corner
        (shape[39][0], shape[39][1]),
        (shape[45][0], shape[45][1]),  # Right eye right corne
        (shape[42][0], shape[42][1]),
        (shape[48][0], shape[48][1]),  # Left Mouth corner
        (shape[54][0], shape[54][1])  # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-300.0, -240.0, -375.0),
        (300.0, -240.0, -375.0),
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (-75.0, 167.0, -135.0),
        (225.0, 170.0, -135.0),  # Right eye right corne
        (75.0, 167.0, -135.0),
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner

    ])

    # Camera internals
    focal_length = 720
    center = (data.cap_height / 2, data.cap_width / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs)

    dst = cv2.Rodrigues(rotation_vector)


    projMatrix = np.zeros((3, 4))
    projMatrix[:3, :3] = dst[0]

    c, r, t, ro, roo, ro7o, euler_angle = cv2.decomposeProjectionMatrix(projMatrix)

    print("euler:", euler_angle)
    return [euler_angle[0][0] + 180, euler_angle[1][0], euler_angle[2][0]]

def calcEyes(ps):
    eyeData = [0, 0, 0, 0]

    unit1_2 = math.pow(helpers.pointDistance(ps[39], ps[42]), 2) / 10
    leftEyeContour = np.array([ps[36], ps[37], ps[38], ps[39], ps[40], ps[41]])
    leftEyeSize = cv2.contourArea(leftEyeContour) / unit1_2
    rightEyeContour = np.array([ps[42], ps[43], ps[44], ps[45], ps[46], ps[47]])
    rightEyeSize = cv2.contourArea(rightEyeContour) / unit1_2

    data.leftEyeAvgSize = (data.leftEyeAvgSize * data.frameCnt_Eye + leftEyeSize) / (data.frameCnt_Eye + 1)
    data.rightEyeAvgSize = (data.rightEyeAvgSize * data.frameCnt_Eye + rightEyeSize) / (data.frameCnt_Eye + 1)
    data.frameCnt_Eye += 1
    # leftEyeAvgSize = data.leftEyeAvgSize
    # rightEyeAvgSize = data.rightEyeAvgSize

    leftEyeAvgSize = 0.43
    rightEyeAvgSize = 0.43

    #print("LeftEyeAvgSize:", leftEyeAvgSize, ", RightEyeAvgSize:", rightEyeAvgSize)

    L_p = leftEyeSize / leftEyeAvgSize
    R_p = rightEyeSize / rightEyeAvgSize

    #print("L_p:", L_p)
    #print("R_p:", R_p)

    if L_p <= 0.7:
        eyeData[0] = 1
    elif L_p >= 2.0:
        eyeData[0] = -0.3
    else:
        eyeData[0] = 1.7 - L_p

    if R_p <= 0.7:
        eyeData[1] = 1
    elif R_p >= 2.0:
        eyeData[1] = -0.3
    else:
        eyeData[1] = 1.7 - R_p

    if eyeData[0] < 0 and eyeData[1] < 0:
        eyeData[2] = eyeData[3] = -(eyeData[0] + eyeData[1]) / 2

    return eyeData


def calcEyebrow(ps, frame):
    unit1 = float(helpers.pointDistance(ps[39], ps[42]))
    d20to40 = helpers.pointDistance(ps[19], ps[39]) / unit1
    d25to43 = helpers.pointDistance(ps[24], ps[42]) / unit1
    eyebrow_d = d20to40 + d25to43

    #print("eyebrowd", eyebrow_d)

    eyebrowData = [0, 0]

    eyebrowData[1] = eyebrow_d * 4 - 7.2

    d22to23 = helpers.pointDistance(ps[21], ps[22]) / unit1

    #print("2223:", d22to23)

    #print("eyebrowData", eyebrowData)

    return eyebrowData

def calcMouth(ps):
    mouthData = [0, 0, 0, 0]
    unit1 = float(helpers.pointDistance(ps[39], ps[42])) / 100

    m2m3d1 = helpers.pointDistance(ps[61], ps[67])
    m2m3d2 = helpers.pointDistance(ps[62], ps[66])
    m2m3d3 = helpers.pointDistance(ps[63], ps[65])
    innerMouthDis = (m2m3d1 + m2m3d2 + m2m3d3) / (3 * unit1)

    m1m4d1 = helpers.pointDistance(ps[49], ps[59])
    m1m4d2 = helpers.pointDistance(ps[50], ps[58])
    m1m4d3 = helpers.pointDistance(ps[51], ps[57])
    m1m4d4 = helpers.pointDistance(ps[52], ps[56])
    m1m4d5 = helpers.pointDistance(ps[53], ps[55])
    outerMouthDis = (m1m4d1 + m1m4d2 + m1m4d3 + m1m4d4 + m1m4d5) / (5 * unit1)
    outerLength = helpers.pointDistance(ps[48], ps[54]) / unit1

    if innerMouthDis <= 5:
        tmp = (outerLength - 120) / 30
        if tmp <= 0:
            mouthData[0] = 0
        elif tmp <= 1:
            mouthData[0] = tmp
        else:
            mouthData[0] = 1
    else:
        openSize = (innerMouthDis - 5) / 50
        if openSize > 1:
            openSize = 1
        if outerLength >= 120:
            mouthData[1] = openSize
        else:
            if openSize * 30 > 1:
                mouthData[2] = 1
            else:
                mouthData[2] = openSize * 30


    return mouthData

def old_calcEye(ps, frame):
    eyeData = [0, 0, 0, 0]

    leftEyeLeftX = ps[36][0]
    leftEyeRightX = ps[39][0]
    leftEyeUpperY = min(ps[37][1], ps[38][1])
    leftEyeLowerY = max(ps[41][1], ps[40][1])

    gray_pixel_cnt = 0

    for j in range(leftEyeLeftX, leftEyeRightX + 1):
        for k in range(leftEyeUpperY, leftEyeLowerY + 1):
            intensity = frame[k][j]
            if max(intensity[0], intensity[1], intensity[2]) - min(intensity[0], intensity[1], intensity[2]) < 35:
                gray_pixel_cnt += 1

    nx = leftEyeRightX - leftEyeLeftX + 1
    ny = leftEyeLowerY - leftEyeUpperY + 1

    g_p = gray_pixel_cnt / float(nx * ny)

    if g_p < 0.2:
        eyeData[0] = 1
        eyeData[2] = 0
    elif g_p < 0.4:
        eyeData[0] = (0.4 - g_p) / 0.2
        eyeData[2] = 0
    elif g_p < 0.8:
        eyeData[0] = eyeData[2] = 0
    else:
        eyeData[0] = 0
        eyeData[2] = (1 - g_p) / 0.2

    rightEyeLeftX = ps[42][0]
    rightEyeRightX = ps[45][0]
    rightEyeUpperY = min(ps[43][1], ps[44][1])
    rightEyeLowerY = max(ps[47][1], ps[46][1])

    gray_pixel_cnt = 0

    for j in range(rightEyeLeftX, rightEyeRightX + 1):
        for k in range(rightEyeUpperY, rightEyeLowerY + 1):
            intensity = frame[k][j]
            if max(intensity[0], intensity[1], intensity[2]) - min(intensity[0], intensity[1], intensity[2]) < 35:
                gray_pixel_cnt += 1

    nx = rightEyeRightX - rightEyeLeftX + 1
    ny = rightEyeLowerY - rightEyeUpperY + 1

    g_p = gray_pixel_cnt / float(nx * ny)

    if g_p < 0.2:
        eyeData[1] = 1
        eyeData[3] = 0
    elif g_p < 0.4:
        eyeData[1] = (0.4 - g_p) / 0.2
        eyeData[3] = 0
    elif g_p < 0.8:
        eyeData[1] = eyeData[3] = 0
    else:
        eyeData[1] = 0
        eyeData[3] = (1 - g_p) / 0.2

    return eyeData
