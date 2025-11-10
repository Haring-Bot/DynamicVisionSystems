import cv2
import numpy as np


minTimestamp = 1.0
maxTimestamp = 2.0
eventLength = 0.1

imageRows = 239
imageCols = 179

data = []

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

def exponential_decay(data):
    imageFunc = np.zeros((imageCols, imageRows))
    normalizedImage = imageFunc * 127
    #print(image)
    print(f"len:{len(data)}")
    biggestTimestamp = data[-1][0]
    #print(biggestTimestamp)

    for event in data:
        if event[3] == 1:
            imageFunc[event[2]-1, event[1]-1] = 1 * np.exp( (event[0] - biggestTimestamp) / eventLength)
        elif event[3] == -1:
            imageFunc[event[2]-1, event[1]-1] = -1 * np.exp( (event[0] - biggestTimestamp) / eventLength)
        else:
            print("ERROR")
            break

    cv2.normalize(imageFunc, normalizedImage, norm_type = cv2.NORM_MINMAX)
    normalizedImage = normalizedImage * 255
    normalizedImage = normalizedImage.astype(np.uint8)
    # print("normal")
    # analyzeImage(imageFunc)
    # print("normalized")
    # analyzeImage(normalizedImage)

    return normalizedImage
        

def showImage(image, freeze = True):
    image = image.astype(np.uint8)
    cv2.imshow("event", image)
    if freeze:
        cv2.waitKey(0)

def saveImage(image, counter = [0]):
    filename = f"results2.2/event_frame_{counter[0]:03d}.png"
    cv2.imwrite(filename, image)
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

def analyzeImage(image):
    highestPixel = image.max()
    lowestPixel = image.min()
    print(f"highest Pixel: {highestPixel}, lowest Pixel: {lowestPixel}")

#print(data[0])

eventData = []

for line in data:
    #print(line[0])
    #print(f"lastCutoff: {lastCutoff}, eventLength: {eventLength}, time: {line[0]}")
    if lastCutoff + eventLength > line[0]:
        #print("appended")
        eventData.append(line)
    else:
        #print("cut off")
        image = exponential_decay(eventData)
        saveImage(image)
        
        eventData = []
        lastCutoff = line[0]