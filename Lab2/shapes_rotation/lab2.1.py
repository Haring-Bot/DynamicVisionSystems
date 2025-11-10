import cv2
import numpy as np


minTimestamp = 1.0
maxTimestamp = 2.0
eventLength = 0.01

imageRows = 239
imageCols = 179

data = []

image = np.ones((imageCols, imageRows))
image = image * 127

print(image.shape)

txtFile = open("events.txt")

lastCutoff = minTimestamp

def event_frame(data):
    imageFunc = np.ones((imageCols, imageRows)) * 127
    #print(f"data: {data}")
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

def saveImage(imageFunc, counter = [0]):
    filename = f"results2.1/event_frame_{counter[0]:03d}.png"
    cv2.imwrite(filename, imageFunc)
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
        #print(lineTransferList)
    elif float(splitLine[0]) > maxTimestamp:
        break
    else:
        continue

#print(data[0])

eventData = []

for line in data:
    #print(line[0])
    #print(f"lastCutoff: {lastCutoff}, eventLength: {eventLength}, time: {line[0]}")
    if lastCutoff + eventLength > line[0]:
        #print("appended")
        eventData.append(line)
    else:
        print("cut off")
        print(len(eventData))
        newImage = event_frame(eventData)
        saveImage(newImage)
        
        eventData = []
        lastCutoff = line[0]