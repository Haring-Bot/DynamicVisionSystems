import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

    return combinedImage, objects

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
    timestampInMicroS = False

    lastTimeframeStart = 0
    timeframe = []
    allTimeframes = []

    if timestampInMicroS == True:
        startTime = startTime * 100000
        endTime = endTime * 100000
        timeStep = timeStep * 100000

    for event in events:
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

    print(f"finished splitting timeframes. Result consits of {len(allTimeframes)} timeframes")
    return allTimeframes


def createSingleImageFromEvents(timestamps, xaddr, yaddr, pol, w, h, startTimestamp, endTimestamp):
    eventframe = []

    for i, timestamp in enumerate(timestamps):

        if timestamp <= startTimestamp:
            continue
        elif timestamp >= endTimestamp:
            break
        elif startTimestamp < timestamp < endTimestamp:
            line = [timestamp, xaddr[i], yaddr[i], pol[i], w, h]
            eventframe.append(line)
        else:
            print("ERROR in timeframe creation, aborting...")
            print(f"startTimestamp: {startTimestamp}, endTimestamp: {endTimestamp}, curTimestamp: {timestamp}")
            break

    return eventframe

def event_frame(data, maxX, maxY):
    imageFunc = np.ones((maxY, maxX)) * 127
    for event in data:
        if event[3] == 1:
            imageFunc[event[2]-1, event[1]-1] = 255
        elif event[3] == 0:
            imageFunc[event[2]-1, event[1]-1] = 0
        else:
            print(f"ERROR: polarity is: {event[3]}")
            break

    return(imageFunc)


def simpleObjectTracker(events, objects, maxX, maxY):
    startTimestamp = 1.0
    timestep = 0.01
    
    segmentedEvents = []
    eventsToAdd = []
    segmentedBlobs = []
    combinedImages = []

    for event in events:
        if event[0] < 1.0:
            continue
        elif event[0] > 5.0:
            break
        
        shortestDistance = 10000
        shortestBlobIndex = 0

        for i, blob in enumerate(objects):
            dX = event[1] - blob[0]
            dY = event[2] - blob[1]
            dTotal = np.sqrt(dX**2 + dY**2)
            if dTotal < shortestDistance:
                shortestDistance = dTotal
                shortestBlobIndex = i
        
        if shortestDistance < objects[shortestBlobIndex][2] / 2:
            objects[shortestBlobIndex][0] = (objects[shortestBlobIndex][0] + event[1]) / 2
            objects[shortestBlobIndex][1] = (objects[shortestBlobIndex][1] + event[2]) / 2
    
        if event[0] < startTimestamp + timestep:
            eventsToAdd.append(event)
        else:
            if len(eventsToAdd) > 0:
                segmentedEvents.append(eventsToAdd)
                segmentedBlobs.append([obj[:] for obj in objects])
            eventsToAdd = [event]
            startTimestamp = event[0]
    
    if len(eventsToAdd) > 0:
        segmentedEvents.append(eventsToAdd)
        segmentedBlobs.append([obj[:] for obj in objects])

    print(f"starting adding images and blobs: {len(segmentedEvents)} segments")

    for eventL, blobs in zip(segmentedEvents, segmentedBlobs):
        image = event_frame(eventL, maxX, maxY)
        keypoints = [cv2.KeyPoint(float(obj[0]), float(obj[1]), float(obj[2])) for obj in blobs]
        combinedImage = cv2.drawKeypoints(image.astype(np.uint8), keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        combinedImages.append(combinedImage)

        plt.imshow(cv2.cvtColor(combinedImage, cv2.COLOR_BGR2RGB))
        plt.title("Detected Blobs")
        plt.axis('off')

    print("finished")
    return(segmentedEvents, segmentedBlobs, combinedImages)

def advancedObjectTracker(events, objects, maxX, maxY):
    chosenBlob = objects[0]
    lastCutoff = 1.0
    interval = 0.01

    relevantX = []
    relevantY = []
    newX = []
    newY = []
    cutoffEvents = []
    combinedImages = []

    for event in events:
        if event[0] < 1.0:
            continue
        elif event[0] > 5.0:
            break
        else:
            pass
        
        dX = event[1] - chosenBlob[0]
        dY = event[2] - chosenBlob[1]
        dTotal = np.sqrt(dX**2 + dY**2)
        if dTotal < chosenBlob[2]/2 and dTotal > 0:
            relevantX.append(dX)
            relevantY.append(dY)
        else:
            continue

        if len(relevantX) > 10 and len(relevantY) > 10:
            del relevantX[0]
            del relevantY[0]
        elif  len(relevantX) <= 10 and len(relevantY) <= 10:
            pass
        else:
            print("!! ERROR !! length of x and y list in advancedObjectTracker is not the same")
        
        xAv = np.mean(relevantX)
        yAv = np.mean(relevantY)

        newX.append(chosenBlob[0] + xAv)
        newY.append(chosenBlob[1] + yAv)

        if event[0] < lastCutoff + interval:
            cutoffEvents.append(event)

        else:
            lastCutoff = event[0]
            image = event_frame(cutoffEvents, maxX, maxY)
            keypoints = [cv2.KeyPoint(float(xCord), float(yCord), 1) for xCord, yCord in zip(newX, newY)]
            combinedImage = cv2.drawKeypoints(image.astype(np.uint8), keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            plt.title("Detected Blobs")
            plt.axis('off')

            newX = []
            newY = []
            cutoffEvents = []
            combinedImages.append(combinedImage)
    return combinedImages

def timeSurfaceTracker(events, objects, maxX, maxY):
    patchSize = 20
    tau = 0.05
    interval = 0.02
    
    timeSurface = np.zeros((maxY, maxX), dtype=np.float32)
    features = []
    
    for pos in objects:
        features.append({
            'x': float(pos[0]),
            'y': float(pos[1]),
            'active': True,
            'trail': [(pos[0], pos[1])]
        })
    
    lastFrameTime = 1.0
    frameEvents = []
    frames = []
    
    for event in events:
        timestamp = event[0]
        
        if timestamp < 1.0:
            continue
        elif timestamp > 5.0:
            break
        
        x, y = event[1], event[2]
        if 0 <= x < maxX and 0 <= y < maxY:
            timeSurface[y, x] = timestamp
        
        for feature in features:
            if not feature['active']:
                continue
            
            dx = x - feature['x']
            dy = y - feature['y']
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < patchSize / 2:
                weight = np.exp(-dist / (patchSize / 3))
                alpha = 0.2 * weight
                
                feature['x'] = (1 - alpha) * feature['x'] + alpha * x
                feature['y'] = (1 - alpha) * feature['y'] + alpha * y
                
                feature['trail'].append((feature['x'], feature['y']))
                if len(feature['trail']) > 30:
                    feature['trail'].pop(0)
        
        frameEvents.append(event)
        
        if timestamp - lastFrameTime >= interval:
            for feature in features:
                if feature['active']:
                    fx, fy = int(feature['x']), int(feature['y'])
                    if 0 <= fx < maxX and 0 <= fy < maxY:
                        timeDiff = timestamp - timeSurface[fy, fx]
                        if timeDiff > 0.15:
                            feature['active'] = False
            
            image = event_frame(frameEvents, maxX, maxY)
            visImage = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
            for feature in features:
                if feature['active']:
                    fx, fy = int(feature['x']), int(feature['y'])
                    
                    radius = patchSize // 2
                    cv2.circle(visImage, (fx, fy), radius, (0, 255, 0), 1)
                    cv2.circle(visImage, (fx, fy), 3, (0, 0, 255), -1)
                    
                    if len(feature['trail']) > 1:
                        pts = np.array([(int(p[0]), int(p[1])) for p in feature['trail']], dtype=np.int32)
                        cv2.polylines(visImage, [pts], False, (255, 165, 0), 2)
            
            frames.append(visImage)
            frameEvents = []
            lastFrameTime = timestamp
    
    print(f"Time surface tracking finished: {len(frames)} frames")
    return frames

def images_to_animation(image_list, interval=100):
    fig, ax = plt.subplots()
    img_plot = ax.imshow(image_list[0])
    ax.axis("off")

    def update(frame_index):
        img_plot.set_data(image_list[frame_index])
        return [img_plot]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(image_list),
        interval=interval,
        blit=True
    )

    return anim



def main():
    root = Path(__file__).parent.parent
    datasetPath = root/"data"

    events, maxX, maxY = eventsParser(os.path.join(datasetPath, "events.txt"))

    image = cv2.imread(os.path.join(datasetPath, "images", "frame_00000022.png"))
    blobsImage, blobObjects = blobDetector(image)
    #none, none, combinedImages = simpleObjectTracker(events, blobObjects, maxX, maxY)
    #combinedImagesAdvanced = advancedObjectTracker(events, blobObjects, maxX, maxY)
    timeSurfaceFrames = timeSurfaceTracker(events, blobObjects, maxX, maxY)

    #animation = images_to_animation(combinedImages, 10)
    #animationCombined = images_to_animation(combinedImagesAdvanced, 10)
    animationTimeSurface = images_to_animation(timeSurfaceFrames, 50)
    plt.show()

if __name__ == "__main__":
    main()