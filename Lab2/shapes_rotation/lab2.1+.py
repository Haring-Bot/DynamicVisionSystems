import cv2
import numpy as np
import os
from pathlib import Path


minTimestamp = 1.0
maxTimestamp = 2.0
eventLength = 0.01

imageRows = 239
imageCols = 179

data = []

image = np.ones((imageCols, imageRows))
image = image * 127

print(image.shape)

txtPath = os.path.join(Path(__file__).parent.parent.parent/"data", "events.txt")
txtFile = open(txtPath)

lastCutoff = minTimestamp

def neighborhoodSuppression(events, tau, R=1):
    timeSurface = np.zeros((imageCols, imageRows))
    lastTimestamp = np.zeros((imageCols, imageRows))
    
    if len(events) == 0:
        return timeSurface
    
    tRef = events[-1][0]
    
    for event in events:
        t = event[0]
        x = event[1] - 1
        y = event[2] - 1
        pol = event[3]
        
        suppressed = False
        for dx in range(-R, R+1):
            for dy in range(-R, R+1):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < imageRows and 0 <= ny < imageCols:
                    if lastTimestamp[ny, nx] > 0 and lastTimestamp[ny, nx] < t:
                        suppressed = True
                        break
            if suppressed:
                break
        
        if not suppressed:
            dt = tRef - t
            timeSurface[y, x] = np.exp(-dt / tau)
            lastTimestamp[y, x] = t
    
    if timeSurface.max() > 0:
        timeSurface = (timeSurface / timeSurface.max() * 255)
    
    return timeSurface

def event_frame(data):
    imageFunc = np.ones((imageCols, imageRows)) * 127
    print(f"image from t={data[0][0]} up to t={data[-1][0]}")
    for event in data:
        if event[3] == 1:
            imageFunc[event[2]-1, event[1]-1] = 255
        elif event[3] == -1:
            imageFunc[event[2]-1, event[1]-1] = 0
        else:
            print("ERROR")
            break

    return(imageFunc)

def showImage(image, freeze = True):
    image = image.astype(np.uint8)
    cv2.imshow("event", image)
    if freeze:
        cv2.waitKey(0)

def saveImage(imageFunc, counter = [0], prefix="eventframe"):
    outputDir = Path(__file__).parent / "results2.1+"
    outputDir.mkdir(exist_ok=True)
    filename = outputDir / f"{prefix}{counter[0]:03d}.png"
    cv2.imwrite(str(filename), imageFunc)
    counter[0] += 1

for line in txtFile:
    lineTransferList = []
    splitLine = line.split(" ")

    if float(splitLine[0]) < maxTimestamp and float(splitLine[0]) > minTimestamp:
        lineTransferList.append(float(splitLine[0]))
        lineTransferList.append(int(splitLine[1]))
        lineTransferList.append(int(splitLine[2]))
        lineTransferList.append((1 if int(splitLine[3]) == 1 else -1))
        data.append(lineTransferList)
    elif float(splitLine[0]) > maxTimestamp:
        break
    else:
        continue

eventData = []

counterRegular = [0]
counterTau1ms = [0]
counterTau10ms = [0]
counterTau100ms = [0]

for line in data:
    if lastCutoff + eventLength > line[0]:
        eventData.append(line)
    else:
        print("cut off")
        print(len(eventData))
        
        newImage = event_frame(eventData)
        saveImage(newImage, counterRegular, "eventframe")
        
        nsImage1ms = neighborhoodSuppression(eventData, 0.001)
        saveImage(nsImage1ms, counterTau1ms, "nstau1ms")
        
        nsImage10ms = neighborhoodSuppression(eventData, 0.01)
        saveImage(nsImage10ms, counterTau10ms, "nstau10ms")
        
        nsImage100ms = neighborhoodSuppression(eventData, 0.1)
        saveImage(nsImage100ms, counterTau100ms, "nstau100ms")
        
        eventData = []
        lastCutoff = line[0]