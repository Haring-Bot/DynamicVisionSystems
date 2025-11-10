import cv2
import numpy as np
import os

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

    print(f"start: {startTimestamp}, end: {endTimestamp}")
    for i, timestamp in enumerate(timestamps):

        if timestamp <= startTimestamp:
            continue
        elif timestamp >= endTimestamp:
            #print(f"break, timestamp: {timestamp}")
            break
        elif startTimestamp < timestamp < endTimestamp:
            line = [timestamp, xaddr[i], yaddr[i], pol[i], h, w]
            eventframe.append(line)
        else:
            print("ERROR in timeframe creation, aborting...")
            print(f"startTimestamp: {startTimestamp}, endTimestamp: {endTimestamp}, curTimestamp: {timestamp}")
            break

    eventframe = event_frame(eventframe)
    return eventframe

def event_frame(data):
    imageFunc = np.ones((data[0][4], data[0][5])) * 127
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
        folder = os.path.join(os.getcwd(), "results4.1")
        os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"event_frame_{counter[0]:03d}.png")
    cv2.imwrite(filename, imageFunc)
    counter[0] += 1

def contrast(params, xs, ys, ts, ps, image_shape):
    t_max = max(ts)
    h_image = np.zeros_like(image_shape, dtype=float)
    maxX, maxY = image_shape.shape
    for i in range(len(xs)):
        x_wraped = xs[i] * params[0] * 1000000
        y_wraped = ys[i] * params[1] * 1000000
        if x_wraped <= maxX and y_wraped <= maxY:
            h_image[y_wraped, x_wraped] += 1
    value = -1 * (sigma**2)*h_image
    return value

def main():
    startTime = 1.0
    endTime = 2.0
    interval = 0.1

    timestamp, x, y, pol, h, w = eventsParser(os.path.join(os.getcwd(), "events.txt"))

    startTimestamp = startTime
    endTimestamp = startTimestamp + interval

    while endTimestamp < endTime:
        print(f"timeframe start: {startTimestamp} end: {endTimestamp}")
        endTimestamp = startTimestamp + interval
        image = createSingleImageFromEvents(timestamp, x, y, pol, h, w, startTimestamp, endTimestamp)
        startTimestamp = endTimestamp
        saveImage(image, "results")



if __name__ == "__main__":
	main()