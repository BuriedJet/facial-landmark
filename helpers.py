import numpy as np
import math
import socket
import data

def sendData():
    address = ('localhost', 9999)
    web = socket.socket()
    web.bind(address)
    web.listen()

    while True:
        conn, addr = web.accept()

        #rd = conn.recv(1024)
        #print(rd)

        data.lock.acquire()
        charData = ''
        for d in data.charData:
            charData += str(d) + '\n'
        data.lock.release()
        #print(charData)
        conn.send(bytes(charData, 'ascii'))

        conn.close()


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def pointDistance(a, b):
    return math.sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]))
