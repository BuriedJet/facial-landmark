def calcAll(points, frame):
    characters = []
    return [0, 0, 0] + calcEyes(points, frame) + [0, 0, 0, 0, 0, 0]


def calcEyes(ps, frame):
    eyeData = [0, 0, 0, 0]

    leftEyeLeftX = ps[36][0]
    leftEyeRightX = ps[39][0]
    leftEyeUpperY = max(ps[37][1], ps[38][1])
    leftEyeLowerY = max(ps[41][1], ps[40][1])

    average_x = float(ps[37][0] + ps[38][0] + ps[40][0] + ps[41][0]) / 4
    average_y = float(ps[37][1] + ps[38][1] + ps[40][1] + ps[41][1]) / 4

    black_pixel_cnt = 1
    gray_pixel_cnt = 0

    for j in range(leftEyeLeftX, leftEyeRightX + 1):
        for k in range(leftEyeUpperY, leftEyeLowerY + 1):
            intensity = frame[k][j]
            if intensity[0] < 30 and intensity[1] < 30 and intensity[2] < 25:
                average_x += j
                average_y += k
                black_pixel_cnt += 1
            if max(intensity[0], intensity[1], intensity[2]) - min(intensity[0], intensity[1], intensity[2]) < 35:
                gray_pixel_cnt += 1

    nx = leftEyeRightX - leftEyeLeftX + 1
    ny = leftEyeLowerY - leftEyeUpperY + 1

    average_x = average_x / float(black_pixel_cnt)
    average_y = average_y / float(black_pixel_cnt)

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

    return eyeData
