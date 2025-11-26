import numpy as np
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
import os

def blobDetector(image):
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 100
    params.filterByCircularity = False
    params.filterByConvexity = False
    
    ver = (cv2.__version__).split(".")
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    blobs = detector.detect(image)
    combinedImage = cv2.drawKeypoints(image, blobs, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.imshow(cv2.cvtColor(combinedImage, cv2.COLOR_BGR2RGB))
    plt.title("Detected Blobs")
    plt.axis('off')
    plt.show()

    objects = []

    for element in blobs:
        objects.append([int(element.pt[0]), int(element.pt[1]), int(element.size)])

    return combinedImage

def eventsParser(path):
    txtFile = open(path)
    events = []
    maxX = 0
    maxY = 0

    for line in txtFile:
        newEvent = []
        splittedLine = line.split(" ")
        newEvent.append(float(splittedLine[0]))      #timestamp
        newEvent.append(int(splittedLine[1]))        #X
        newEvent.append(int(splittedLine[2]))        #Y
        newEvent.append(int(splittedLine[3]))        #polarity
        if int(newEvent[1]) > maxX:
            maxX = int(newEvent[1])
        if int(newEvent[2]) > maxY:
            maxY = int(newEvent[2])
                       
        events.append(newEvent)
                       
    return events, maxX, maxY

def splitEvents(events):
    startTime = 1.0
    endTime = 5.0
    timeStep = 0.001
    timestampInUs = False

    lastTimeframeStart = 0
    timeframe = []
    allTimeframes = []

    if timestampInUs == True:
        startTime = startTime * 100000
        endTime = endTime * 100000
        timeStep = timeStep * 100000

    for event in events:
        print(event)
        timestamp = event[0]
        if timestamp < startTime:
            continue
        elif timestamp < lastTimeframeStart + startTime:
            timeframe.append(event)
        elif timestamp > lastTimeframeStart + startTime:
            allTimeframes.append(timeframe)
            lastTimeframeStart = timestamp
        elif timestamp > endTime:
            break
        else:
            print("ERROR in event splitter")

    print(f"finished splitting timeframes. Result consits of {allTimeframes.size}")
    return allTimeframes


def createSingleImageFromEvents(timestamps, xaddr, yaddr, pol, w, h, startTimestamp, endTimestamp):
    eventframe = []

    #print(f"start: {startTimestamp}, end: {endTimestamp}")
    for i, timestamp in enumerate(timestamps):

        if timestamp <= startTimestamp:
            continue
        elif timestamp >= endTimestamp:
            #print(f"break, timestamp: {timestamp}")
            break
        elif startTimestamp < timestamp < endTimestamp:
            line = [timestamp, xaddr[i], yaddr[i], pol[i], w, h]
            eventframe.append(line)
        else:
            print("ERROR in timeframe creation, aborting...")
            print(f"startTimestamp: {startTimestamp}, endTimestamp: {endTimestamp}, curTimestamp: {timestamp}")
            break

    return eventframe

def main():
    root = Path(__file__).parent.parent
    datasetPath = root/"data"

    events = eventsParser(os.path.join(datasetPath, "events.txt"))
    timeframes = splitEvents(events)

    image = cv2.imread(os.path.join(datasetPath, "images", "frame_00000022.png"))
    blobs = blobDetector(image)

if __name__ == "__main__":
    main()