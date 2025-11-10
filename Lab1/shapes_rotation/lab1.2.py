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
plt.show()


### ANSWERS ###
# How long is the sequence in exc 1.1?
#   1 second
# 
# Whats the resolution of the event timestamp?
#   due to smallest difference between indiv. timestamps 1ns. Probably more but for this refer to datasheet. 
#
# What does the time difference between consectutive events depend on
#   sensor max frequency and occurence of activities
#
# What does positive and negative event polarity mean?
#   positive: increasing brightness on pixel
#   negative: decreasing brightness on pixel
#
# In which direction are the objects moving?
#   postive X