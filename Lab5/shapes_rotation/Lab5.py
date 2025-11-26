import cv2
import numpy as np
import os
import statistics
from scipy.optimize import fmin, minimize
from datetime import datetime

def eventsParser(path):
    timestamps = []
    x = []
    y = []
    pol = []
    maxX = 0
    maxY = 0

    txtFile = open(path)

    for line in txtFile:
        splittedLine = line.split(" ")
        timestamps.append(float(splittedLine[0]))
        x.append(int(splittedLine[1]))
        y.append(int(splittedLine[2]))
        pol.append(int(splittedLine[3]))
        if int(splittedLine[1]) > maxX:
            maxX = int(splittedLine[1])
        if int(splittedLine[2]) > maxY:
            maxY = int(splittedLine[2]
)
    return timestamps, x, y, pol, maxX, maxY

def createSingleImageFromEvents(timestamps, xaddr, yaddr, pol, w, h, startTimestamp, endTimestamp):
    eventframe = []

    for i, timestamp in enumerate(timestamps):
        if timestamp <= startTimestamp:
            continue
        elif timestamp >= endTimestamp:
            break
        elif startTimestamp < timestamp < endTimestamp:
            line = [timestamp, xaddr[i], yaddr[i], pol[i], w, h]  # Changed: w, h order to match parameters
            eventframe.append(line)
        else:
            print("ERROR in timeframe creation, aborting...")
            print(f"startTimestamp: {startTimestamp}, endTimestamp: {endTimestamp}, curTimestamp: {timestamp}")
            break

    return eventframe

def event_frame(data):
    imageFunc = np.ones((data[0][5], data[0][4])) * 127
    #print(f"data: {data}")
    #print(f"image from t={data[0][0]} up to t={data[-1][0]}")
    for event in data:
        if event[3] == 1:
            imageFunc[event[2]-1, event[1]-1] = 255
        elif event[3] == 0:
            imageFunc[event[2]-1, event[1]-1] = 0
        else:
            print(f"ERROR: polarity is: {event[3]}")
            break

    return(imageFunc)

def saveImage(imageFunc, folder = None, counter = [0]):
    if folder == None:
        folder = os.path.join(os.getcwd(), "results5")
    else:
        folder = os.path.join(os.getcwd(), folder) 
    
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"event_frame_{counter[0]:03d}.png")
    cv2.imwrite(filename, imageFunc)
    counter[0] += 1
    print(f"image saved as {filename}")

def contrast(params, xs, ys, ts, ps, image_shape):
    theta_x, theta_y = params * 1000000
    t_max = np.max(ts)

    h_image = np.zeros(image_shape, dtype=float)
    maxY, maxX = image_shape

    for x, y, t in zip(xs, ys, ts):
        dt = t - t_max
        xw = x - dt * theta_x
        yw = y - dt * theta_y

        xw = int(round(xw))
        yw = int(round(yw))

        if 0 <= xw < maxX and 0 <= yw < maxY:
            h_image[yw, xw] += 1

    return -np.var(h_image)

def rectifyImage(xs, ys, ts, ps, vX, vY, imageShape):
    print(f"image is being rectified by x={vX*1000000 / 10}, y={vY*1000000 / 10} p/s")
    
    tMin = np.min(ts)
    rImage = np.zeros(imageShape, dtype=float)
    for x, y, t, p in zip(xs, ys, ts, ps):
        dt = t - tMin
        dX = vX * dt * 1000000
        dY = vY * dt * 1000000
        rX = int(round(x - dX))
        rY = int(round(y - dY))
        if 0 <= rX < imageShape[1] and 0 <= rY < imageShape[0]:
            if p == 0:
                rImage[rY, rX] = 0
            elif p == 1:
                rImage[rY, rX] = 255
            else:
                print(f"!ERROR in image rectification! value is {p}")
    return rImage

def main():
    startTime = 1.0
    endTime = 2.0
    interval = 0.1

    timestamp, x, y, pol, maxX, maxY = eventsParser(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "events.txt")) 

    durationFminL = []
    durationMinimizeL = []

    startTimestamp = startTime
    endTimestamp = startTimestamp + interval

    while endTimestamp < endTime:
        endTimestamp = startTimestamp + interval
        eventframe = createSingleImageFromEvents(timestamp, x, y, pol, maxX, maxY, startTimestamp, endTimestamp)  
        saveImage(event_frame(eventframe), "resultsBefore")

        xs = [event[1] for event in eventframe]
        ys = [event[2] for event in eventframe]
        ts = [event[0] for event in eventframe]
        ps = [event[3] for event in eventframe]
        
        startTime = datetime.now()

        args = (xs, ys, ts, ps, (maxY, maxX)) 
        argmax = fmin(contrast, (0, 0), args=args, disp=False)

        timeFmin = datetime.now()
        durationFminL.append((timeFmin - startTime).total_seconds())

        argsMin = (xs, ys, ts, ps, (maxY, maxX))
        argmaxMin = minimize(contrast, [0, 0], args=argsMin, method='Powell')
        timeMinimize = datetime.now()
        durationMinimizeL.append((timeMinimize - timeFmin).total_seconds())

        eventframe = rectifyImage(xs, ys, ts, ps, argmax[0], argmax[1], (maxY, maxX)) 
        eventframe2 = rectifyImage(xs, ys, ts, ps, argmaxMin.x[0], argmaxMin.x[1], (maxY, maxX)) 

        image = eventframe.astype(np.uint8)
        image2 = eventframe2.astype(np.uint8)
        saveImage(image, "resultsFmin")
        saveImage(image2, "resultsPowell")
        startTimestamp = endTimestamp

    #average fmin: 3.44 average Powell: 8.09
    print(f"average fmin: {statistics.mean(durationFminL)}, Powell: {statistics.mean(durationMinimizeL)}")


if __name__ == "__main__":
	main()