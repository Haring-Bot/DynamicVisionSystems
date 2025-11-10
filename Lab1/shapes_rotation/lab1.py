import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

txtFile = open("events.txt")
myList =[]
myTime = []
myX = []
myY = []
myIntensity = []

for line in txtFile:
    myList.append(line)
    #print(line)

for line in myList:
    mySingleLineList = []
    mySingleLineList = line.split(" ")
    myTime.append(float(mySingleLineList[0]))
    myX.append(int(mySingleLineList[1]))
    myY.append(int(mySingleLineList[2]))
    myIntensity.append(int(mySingleLineList[3]))
    if float(mySingleLineList[0]) > 1:
        break
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
plt.show()