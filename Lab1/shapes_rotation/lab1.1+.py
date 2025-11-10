import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

minTimestamp = 0.5
maxTimestamp = 1.0
#maxAmountEvents = 8000
maxAmountEvents = 10000000

txtFile = open("events.txt")
myList =[]
myTime = []
myX = []
myY = []
myIntensity = []


def summarizeMovement(listTime, listX, listY, interval):
    lastcutoff = 0
    timeframeX, timeframeY = [], []
    positionSummaryX, positionSummaryY = [], []
    movementSummaryX, movementSummaryY = [], []
    lastX, lastY = 0, 0

    for i, timestamp in enumerate(listTime):
        if timestamp < lastcutoff + interval:
            timeframeX.append(listX[i])
            timeframeY.append(listY[i])
        else:
            lastcutoff = lastcutoff + interval
            print(f"timestamp in moment of cut {timestamp:.6f}")  # Show 6 decimal places
            print(f"lastCutoff {lastcutoff}")
            print(f"length timeframe: {len(timeframeX)}, {len(timeframeY)}")
            meanX = np.mean(timeframeX)
            meanY = np.mean(timeframeY)

            positionSummaryX.append(meanX)
            positionSummaryY.append(meanY)
            movementSummaryX.append(meanX - lastX)
            movementSummaryY.append(meanY - lastY)

            lastX = meanX
            lastY = meanY

            timeframeX, timeframeY = [], []

    #print(f"average position into x-direction: {positionSummaryX}, y-direction: {positionSummaryY}")

    xAxis = np.linspace(0, listTime[-1], len(positionSummaryX))
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10, 10))

    ax1.plot(xAxis, positionSummaryX, label= "x-pos")
    ax1.set_title("x-position")

    ax2.plot(xAxis, positionSummaryY, label= "y-pos")
    ax2.set_title("y-position")

    ax3.plot(xAxis, movementSummaryX, label="x-movement")
    ax3.set_title("x-movement")

    ax4.plot(xAxis, movementSummaryY, label="y-movement")
    ax4.set_title("y-movement")

    plt.tight_layout()


for line in txtFile:
    myList.append(line)
    #print(line)

amountEvents = 0

for line in myList:
    mySingleLineList = []
    mySingleLineList = line.split(" ")
    if float(mySingleLineList[0]) < minTimestamp:
        pass
    elif float(mySingleLineList[0]) > maxTimestamp:
        break
    elif amountEvents > maxAmountEvents:
        break
    else:
        myTime.append(float(mySingleLineList[0]))
        myX.append(int(mySingleLineList[1]))
        myY.append(int(mySingleLineList[2]))
        myIntensity.append(int(mySingleLineList[3]))
        amountEvents += 1
        #print(mySingleLineList[0])

print(f"number of events: {len(myList)}")
print(f"first timestamp at: {myTime[0]}, last timestamp at: {myTime[-1]}")
print(f"largest x-cord: {max(myX)} and largest y-cord: {max(myY)}")
print(f"amount of 0 intensity: {myIntensity.count(0)}, compared to amount of 1 intensity: {myIntensity.count(1)}")

posIntensityList = []
negIntensityList = []

for i, event in enumerate(myTime):
    if myIntensity[i] == 1:
        posIntensityList.append([myTime[i], myX[i], myY[i], myIntensity[i]])
    elif myIntensity[i] == 0:
        negIntensityList.append([myTime[i], myX[i], myY[i], myIntensity[i]])
    else:
        print("Error in Intensity sorting")

posTimes = [event[0] for event in posIntensityList]
posX = [event[1] for event in posIntensityList]
posY = [event[2] for event in posIntensityList]

negTimes = [event[0] for event in negIntensityList]
negX = [event[1] for event in negIntensityList]
negY = [event[2] for event in negIntensityList]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d") 

ax.scatter(posX, posY, posTimes, c="r", marker="o", s=2, label="Positive Events")
ax.scatter(negX, negY, negTimes, c="b", marker="o", s=2, label="Negative Events")

ax.set_title("Events (Red = Positive, Blue = Negative)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Time")

ax.legend()

plt.tight_layout()

summarizeMovement(myTime, myX, myY, 0.01)

plt.show()