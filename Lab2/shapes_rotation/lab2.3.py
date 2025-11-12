import cv2
import numpy as np


minTimestamp = 1.0
maxTimestamp = 2.0
eventLength = 0.01

imageRows = 239
imageCols = 179

data = []

image = np.ones((imageRows, imageCols))
image = image * 127

print(image.shape)

txtFile = open("events.txt")

lastCutoff = minTimestamp

#check GPT
#currently for goes through all values of one category, not through all events
def event_frame(data, image):
    #print(f"data: {data}")
    print(f"image from t={data[0][0]} up to t={data[-1][0]}")
    for event in data:
        if event[3] == 1:
            image[event[2]-1, event[1]-1] = 255
        elif event[3] == -1:
            image[event[2]-1, event[1]-1] = 0
        else:
            print("ERROR")
            break

    return(image)

def exponential_decay(data, image):
    image = np.zeros_like(image, dtype=float)
    normalizedImage = image * 0
    #print(image)
    
    biggestTimestamp = data[-1][0]
    #print(biggestTimestamp)

    for event in data:
        if event[3] == 1:
            image[event[2]-1, event[1]-1] = 255 * np.exp( (event[0] - biggestTimestamp) / eventLength)
        elif event[3] == -1:
            image[event[2]-1, event[1]-1] = 0
        else:
            print("ERROR")
            break

    # cv2.normalize(image, normalizedImage, norm_type = cv2.NORM_MINMAX)
    # normalizedImage = normalizedImage.astype(np.uint8)
    normalizedImage = image

    return normalizedImage

def event_frequency(data, image):
    counts = np.zeros_like(image, dtype=float)

    for event in data:
        ts, x, y, p = event[0], event[1], event[2], event[3]
        if p == 1:
            counts[x-1, y-1] += 1
        elif p == -1:
            continue
        else:
            print("ERROR: unexpected polarity", p)
            break

    if counts.max() > 0:
        normalized = cv2.normalize(counts, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
    else:
        normalized = np.zeros_like(counts, dtype=np.uint8)

    return normalized
        

def showImage(image, freeze = True):
    image = image.astype(np.uint8)
    cv2.imshow("event", image)
    if freeze:
        cv2.waitKey(0)

def saveImage(image, counter = [0]):
    filename = f"results2.3/event_frame_{counter[0]:03d}.png"
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
        image = event_frequency(eventData, image)
        saveImage(image)
        
        eventData = []
        lastCutoff = line[0]
