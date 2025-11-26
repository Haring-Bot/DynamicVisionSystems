import os
import numpy as np
from pprint import pprint
import cv2
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
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
    print(f"loaded dataset from {path}")

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

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        classes = sorted(os.listdir(root_dir))
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            cls_folder = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_folder):
                self.images.append(os.path.join(cls_folder, img_name))
                self.labels.append(idx)
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class customConv(nn.Module):
    def __init__(self, inputCh, outputCh, kernelSize):
        super(customConv, self).__init__()
        self.conv = nn.Conv2d(inputCh, outputCh, kernelSize, padding = 1)
        self.bn = nn.BatchNorm2d(outputCh)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        
        return x
    

class customCNN(nn.Module):
    def __init__(self, numClasses):
        super(customCNN, self).__init__()
        self.layer1 = customConv(3, 32, 3)
        self.layer2 = customConv(32, 32, 3)
        self.layer3 = customConv(32, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, numClasses)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def trainModel(path, testPath):
    device = "cpu"

    print(torch.__version__)
    print("starting training")

    transform = transforms.Compose([
        transforms.Resize((34, 34)),
        transforms.RandomHorizontalFlip(p=0.5),     #only for extension
        transforms.RandomRotation(10),              #only for extension
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create train and validation datasets
    train_dataset = CustomImageDataset(root_dir=path, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = CustomImageDataset(root_dir=testPath, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = customCNN(numClasses=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # History tracking
    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    best_val_accuracy = 0.0
    nEpochs = 5
    for epoch in range(nEpochs):
        print(f"starting training epoch {epoch+1} out of {nEpochs}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_dataloader)
        train_accuracy = 100 * correct / total
        history['loss'].append(train_loss)
        history['accuracy'].append(train_accuracy)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_total = 0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_dataloader)
        val_accuracy = 100 * val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f"Epoch [{epoch+1}/{nEpochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(os.getcwd(), "results", "best_model.pth")
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"Best model saved with validation accuracy: {val_accuracy:.2f}%")

    # Save final model
    final_model_path = os.path.join(os.getcwd(), "results", "final_model.pth")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save({
        'epoch': nEpochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Plot training history
    plot_training_history(history)
    
    return model, history


def plot_training_history(history):
    """Plot training and validation accuracy and loss"""
    
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='train')
    plt.plot(history['val_accuracy'], label='test')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='test')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(os.getcwd(), "results", "training_history.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    
    plt.show()

def predict_single_image(model, image_path, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((34, 34)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    original_image = Image.open(image_path).convert("RGB")
    
    image = transform(original_image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        probabilities = F.softmax(model(image), dim=1).cpu().numpy()[0]
    
    plt.imshow(original_image)
    plt.title(f"Input Image: {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()

    print(f"\nPrediction for: {image_path}")
    for x in range(10):
        print(f"The probability that the number is {x} equals {probabilities[x]*100:.2f}%")
    
    predicted_class = np.argmax(probabilities)
    print(f"\nPredicted: {predicted_class} ({probabilities[predicted_class]*100:.2f}%)")
    
    return probabilities, predicted_class

def main():
    path = os.path.join(os.getcwd(), "raw_data")
    print(path)

    savedPath = "/home/julian/Documents/FH/Krakow/DynamicVisionSystems/Lab4/results/dataset.pkl"
    imagesPathTrain = "/home/julian/Documents/FH/Krakow/DynamicVisionSystems/Lab4/images/train/images"
    imagesPathTest = "/home/julian/Documents/FH/Krakow/DynamicVisionSystems/Lab4/images/test/images"
    
    train_new_model = False
    
    if train_new_model:
        data = loadDataset(savedPath)
        print(type(data))
        
        model, history = trainModel(imagesPathTrain, imagesPathTest)
        print("History keys:", history.keys())
    else:
        model = customCNN(numClasses=10)
    
    best_model_path = os.path.join(os.getcwd(), "results", "best_model.pth")
    checkpoint = torch.load(best_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model (Val Acc: {checkpoint['val_accuracy']:.2f}%)")
    
    test_image_path = os.path.join(imagesPathTest, "3/event_frame_299489.png")
    probabilities, predicted_class = predict_single_image(model, test_image_path, device='cpu')

#epoch= amount of times model is run and weights updated
#batch_size = amount of images being processed at once

if __name__ == "__main__":
    main()