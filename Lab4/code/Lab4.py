import os
import numpy as np
from pprint import pprint
import cv2
import pickle
import torch

# average amount of train events for class 3: 4703.4139618333065
# average amount of train events for class 1: 2432.0760901809554
# average amount of train events for class 6: 4215.370057451842
# average amount of train events for class 4: 3794.4542964738102
# average amount of train events for class 5: 4372.011805939864
# average amount of train events for class 2: 4708.0715005035245
# average amount of train events for class 7: 3687.1762170790103
# average amount of train events for class 8: 4701.756110066655
# average amount of train events for class 0: 5443.992064832011
# average amount of train events for class 9: 3927.132795427803

# average amount of test events for class 3: 4693.830693069307
# average amount of test events for class 1: 2411.0977973568283
# average amount of test events for class 6: 4370.312108559499
# average amount of test events for class 4: 3819.051934826884
# average amount of test events for class 5: 4458.270179372197
# average amount of test events for class 2: 4709.3071705426355
# average amount of test events for class 7: 3683.0262645914395
# average amount of test events for class 8: 4782.366529774127
# average amount of test events for class 0: 5400.781632653061
# average amount of test events for class 9: 4011.8037661050544

# amount of train images: 60000, test images: 10000, proportion: 0.16666666666666666



def analyzeDataset(path):
    testPath = os.path.join(path, "Test")
    trainPath = os.path.join(path, "Train")
    nTrain = 0
    nTest = 0
    nTrainClass = 0
    nTestClass = 0
    totalEventsTrain = 0
    totalEventsTest = 0
    totalImagesTrain = {}
    totalImagesTest = {}
    imagesClass = []

    if os.path.exists(testPath) and os.path.exists(trainPath):
        print("both paths exits... continuing")
    else:
        print("path missing... shutting down")

    print(trainPath)
    for folder in os.listdir(trainPath):
        for file in os.listdir(os.path.join(trainPath, folder)):
            nTrain += 1
            #nTrainClass += 1
            timestamps, xaddr, yaddr, pol, h, w = read_dataset(os.path.join(trainPath, folder, file))
            #totalEventsTrain += (xaddr).size
            imagesClass.append(createImageFromEvents(timestamps, xaddr, yaddr, pol, h, w))
        #print(f"average amount of test events for class {folder}: {totalEventsTest / nTestClass}")
        nTrainClass = 0
        totalEventsTrain = 0
        transferImages = []
        for images in imagesClass:
            for individualImage in images:
                transferImages.append(individualImage)
        totalImagesTrain[folder] = transferImages
        imagesClass = []
        print(f"finished {folder}")

    for folder in os.listdir(testPath):
        for file in os.listdir(os.path.join(testPath, folder)):
            nTest += 1
            #nTestClass += 1
            timestamps, xaddr, yaddr, pol, h, w = read_dataset(os.path.join(testPath, folder, file))
            #totalEventsTest+= (xaddr).size
            imagesClass.append(createImageFromEvents(timestamps, xaddr, yaddr, pol, h, w))
        #print(f"average amount of test events for class {folder}: {totalEventsTest / nTestClass}")
        nTestClass = 0
        totalEventsTest = 0
        transferImages = []
        for images in imagesClass:
            for individualImage in images:
                transferImages.append(individualImage)
        totalImagesTest[folder] = transferImages
        imagesClass = []
        print(f"finished {folder}")

    print(f"amount of train images: {nTrain}, test images: {nTest}, proportion: {nTest/ nTrain}")
    return(totalImagesTrain, totalImagesTest)

def read_dataset(filename):
	f = open(filename, 'rb')
	raw_data = np.fromfile(f, dtype=np.uint8)
	f.close()
	raw_data = np.uint32(raw_data)
	all_y = raw_data[1::5]
	all_x = raw_data[0::5]
	all_p = (raw_data[2::5] & 128) >> 7 #bit 7
	all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
	time_increment = 2 ** 13
	overflow_indices = np.where(all_y == 240)[0]
	for overflow_index in overflow_indices:
		all_ts[overflow_index:] += time_increment
	td_indices = np.where(all_y != 240)[0]
	x = all_x[td_indices]
	w = x.max() + 1
	y = all_y[td_indices]
	h = y.max() + 1
	ts = all_ts[td_indices]
	p = all_p[td_indices]
	return ts, x, y, p, h, w

def createImageFromEvents(timestamps, xaddr, yaddr, pol, h, w):
    eventLength = 50000
    lastCutoff = 0
    event = []
    listEvents = []
    images = []

    for i, timestamp in enumerate(timestamps):
        if timestamp < lastCutoff + eventLength:
            line = [timestamp, xaddr[i], yaddr[i], pol[i], h, w]
            #print(line)
            event.append(line)
        else:
            listEvents.append(event)
            event = []
            lastCutoff = timestamp
            #print(timestamp)

    if listEvents:
        listEvents.pop(-1)
    #print("finished element")

    #pprint(listEvents)
    for event in listEvents:
        images.append(event_frame(event))

    return images
             
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

def saveImage(imageFunc, folder = None,counter = [0]):
    if folder == None:
        folder = os.path.join(os.getcwd(), "results4.1")
        os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"event_frame_{counter[0]:03d}.png")
    cv2.imwrite(filename, imageFunc)
    counter[0] += 1

def saveDataset(filename, data):
    with open(filename, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"data saved as {filename}")

def loadDataset(path):
    with open(path, "rb") as file:
        data = pickle.load(file)
    print(f"loaded dataset from f{path}")

    return data

def prepareData(path):
    allImagesTrain, allImagesTest = analyzeDataset(path)
    allImagesTotal = {"train" : allImagesTrain, "test" : allImagesTest}
    try: 
        print(f"size total images list train: {len(allImagesTrain)}")
        print(f"size total images list test: {len(allImagesTest)}")
    except:
        print("size print didnt work")

    savePath = os.path.join(os.getcwd(), "results/dataset.pkl")
    saveDataset(savePath, allImagesTotal)

    return savePath

def trainModel(data):
    print(torch.__version__)
    print("starting training")
    


def main():
    path = os.path.join(os.getcwd(), "raw_data")
    print(path)

    savedPath = "/home/julian/Documents/FH/Krakow/dynamicVisionSensors/Lab4/results/dataset.pkl"
    #savedPath = prepareData(path)

    data = loadDataset(savedPath)
    trainModel(data)
    

    # for folderName, content in allImagesTotal.items():
    #     os.makedirs(os.path.join(os.getcwd(), folderName, "images"), exist_ok = True)
    #     for label, data in content.items():
    #         currentFolder = os.path.join(os.getcwd(), folderName, "images", label)
    #         os.makedirs(currentFolder, exist_ok=True)
    #         for image in data:
    #             saveImage(image, folder = currentFolder)
    #         print(f"finished saving images in {currentFolder}")



    

if __name__ == "__main__":
    main()