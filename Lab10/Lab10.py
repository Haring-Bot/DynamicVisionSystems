import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
        elif event[3] == -1:
            imageFunc[event[2]-1, event[1]-1] = 0
        else:
            print(f"ERROR: polarity is: {event[3]}")
            break

    return(imageFunc)

def matchTimestampImage(timestamp):
    txtFile = open("reconstruction/city_09d_150_200/timestamps.txt")
    for line in txtFile:
        singleLine = []
        singleLine = line.split(" ")
        if float(singleLine[1]) == timestamp:
            txtFile.close()
            return(singleLine[0])
    
    txtFile.close()
    return(-1)

def images_to_animation(image_pairs, interval=100):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    img_plot1 = ax1.imshow(image_pairs[0][0], cmap='gray')
    ax1.set_title('Event Frame')
    ax1.axis("off")
    
    img_plot2 = ax2.imshow(image_pairs[0][1], cmap='gray')
    ax2.set_title('HDR Reconstruction')
    ax2.axis("off")
    
    plt.tight_layout()

    def update(frame_index):
        img_plot1.set_data(image_pairs[frame_index][0])
        img_plot2.set_data(image_pairs[frame_index][1])
        return [img_plot1, img_plot2]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(image_pairs),
        interval=interval,
        blit=True
    )

    return anim

def main():
    # Load MATLAB v5 file using scipy
    matData = scipy.io.loadmat("city_09d_150_200.mat")
    
    # Extract data
    imagesOrginal = matData['image']
    timestampsImagesOrginal = matData['time_image'].flatten() / 1000000
    
    events = matData['events']
    timestamps = events[:, 0] / 1000000
    xValues = events[:, 1].astype(int)
    yValues = events[:, 2].astype(int)
    pol = events[:, 3].astype(int)

    pairs = []
    startTimestamp = 0

    for individualTimestampImage in timestampsImagesOrginal:
        mask = (timestamps > startTimestamp) & (timestamps < individualTimestampImage)
        eventTimestamps = timestamps[mask]
        eventX = xValues[mask]
        eventY = yValues[mask]
        eventPol = pol[mask]
        
        startTimestamp = individualTimestampImage
        print(individualTimestampImage)
        print(f"Collected {np.sum(mask)} events")

        eventframe = createSingleImageFromEvents(eventTimestamps, eventX, eventY, eventPol, 640, 480, 0, 100)
        event_image = event_frame(eventframe, 640, 480)

        matchingImage = matchTimestampImage(individualTimestampImage)
        if matchingImage == -1:
            print("!!ERROR matching!!")
            continue
        
        hdr_path = f"reconstruction/city_09d_150_200/{matchingImage}"
        if os.path.exists(hdr_path):
            import cv2
            hdr_image = cv2.imread(hdr_path, cv2.IMREAD_GRAYSCALE)
            
            pair = [event_image, hdr_image]
            print(f"timestamp: {individualTimestampImage}, match: {matchingImage}")
            pairs.append(pair)
        else:
            print(f"HDR image not found: {hdr_path}")
    
    print(f"\nCreating animation with {len(pairs)} frames...")
    anim = images_to_animation(pairs, interval=10)
    plt.show()


if __name__ == "__main__":
    main()