import platform
import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


# loaded the data without transforming to tensors and printed their graphs
train_dataset = torchvision.datasets.CIFAR10(
     root='..', 
     train=True,
     download=False)

test_dataset = torchvision.datasets.CIFAR10(
     root='..', 
     train=False, 
     download=False)

# printed the lenghts of data sets 

X  = len(train_dataset)
Y = len(test_dataset)

print(X)
print(Y)

# Histogram graph
def plot_histogram1(labels, dataset_name):
    labels = np.array(labels)
    C,counts = np.unique(labels, return_counts=True)
    plt.bar(C, counts)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title(f'{dataset_name} ')
    plt.show()

train_label = [label for images, label in train_dataset]
plot_histogram1(train_label, 'training')

def plot_histogram2(labels, dataset_name):
    labels = np.array(labels)
    C,counts = np.unique(labels, return_counts=True)
    plt.bar(C, counts)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title(f' {dataset_name}' )
    plt.show()

test_label = [label for images, label in test_dataset]
plot_histogram2(test_label, 'test')


# print(train_dataset)
# print(test_dataset)
# print(train_loader)
# print(test_loader)


# here I have converted datasets to tensor using transforms.ToTensor() which does [ C , H , W ] --> [ H , W , C]
train_dataset = torchvision.datasets.CIFAR10(
     root='..', 
     train=True,
     transform=transforms.ToTensor(),
     download=False)

test_dataset = torchvision.datasets.CIFAR10(
     root='..', 
     train=False,
     transform=transforms.ToTensor(), 
     download=False)

classes = train_dataset.classes
print("classes",classes)


# Here I have printed each images form datasets by accesing from train data sets which accesses labels and images by unpacking
i = torch.randint(0,len(train_dataset),size=[1]).item()
images,labels = train_dataset[i]
print("image,labels : ",images,labels)
plt.figure(figsize=(5,5))
plt.imshow(np.transpose(images) )
plt.title(f"Label: {labels,classes[labels]}")
plt.grid(False)  
plt.show()

# printed dimensions of images and respective classes
print(f"image shape is {images.shape}")
print(f"image label is {classes[labels]}")


# Here I have splitted datasets and accesses their data under "train,valid,test" loader adn seperated data intp 80% for training and 20%b for validation
Batch_size = 10

train_datasets, valid_datasets = torch.utils.data.random_split(train_dataset, [int(0.8 * len(train_dataset)), (len(train_dataset) - int(0.8 * len(train_dataset)))])

train_loader = torch.utils.data.DataLoader(
     dataset=train_datasets,
     batch_size=Batch_size,
     shuffle=True)

valid_loader = torch.utils.data.DataLoader(
     dataset=valid_datasets,
     batch_size=Batch_size,
     shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size = Batch_size,
    shuffle=True)

print("train_loader has 80% of  data")
print("valid_loader has 20% of  data")

print(f"length of train_loader : {len(train_loader)} which is 80% with batch size : {Batch_size}")
print(f"length of valid_loader : {len(valid_loader)} which is 20% with batch_size : {Batch_size}")

images,labels = next(iter(train_loader))
images.shape,labels.shape

if platform.system() == 'Darwin':
   DEVICE =  torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


if platform.system() == 'Darwin':
   DEVICE =  torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class ConvNet(nn.Module):
    def __init__(self, num_classes = len(classes)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 5 * 5 ,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16 * 5 * 5) # we can also use flatten(x) what this x.view does is work of reshaping
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = ConvNet()
model.to(DEVICE)
print(model)



# train the model for train_loss , valid_loss,  test_loss
learning_rate = 0.005
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


torch.manual_seed(42)
import time
epoches = 1
start = time.time()


correct_validation = 0

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

train_loss = []
valid_loss = []
test_loss  = []
correct_test = 0
for epoch in tqdm(range(epoches)):
    print(f"Epoch : {epoch}")
    T_loss = 0
    V_loss = 0
    Te_loss = 0
    for i , (images,labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        model.train()

        #forward_pass
        train_Output = model(images)
        loss = loss_fn(train_Output,labels)
        T_loss = T_loss + loss.item()
        train_loss.append(T_loss)

        #backward propogation
        optimizer.zero_grad()
        loss.backward() # used to do the or access loss 
        optimizer.step() #  used to update the parameters 
        
        # calculated 
        if (i+1) % 300 == 0:
               _, predicted = torch.max(train_Output, 1)
               train_correct = (predicted == labels).sum().item()
               accuracy_train = (train_correct / Batch_size)* 100
               print(f"epoch [{epoch+1}/{epoches}],step [{i+1}/{len(train_loader)}],train_loss : {T_loss}, Accuracy : {accuracy_train}%")
        
    torch.save(model.state_dict(), 'RAHUL_SHAH1.pt') # saved my model under pt format file

    print("proccess ended for training")
    model.eval()
    for images , labels in (valid_loader):
       images = images.to(DEVICE)
       labels = labels.to(DEVICE)

       #forward_pass
       valid_Output = model(images)
       loss_1 = loss_fn(valid_Output,labels)
       V_loss = V_loss + loss_1.item()
       valid_loss.append(V_loss)

       print(f"valid_loss : {V_loss}")
      
        
    model.eval()
    for images , labels in (test_loader):
       images = images.to(DEVICE)
       labels = labels.to(DEVICE)

       #forward_pass
       test_Output = model(images)
       loss_2 = loss_fn(test_Output,labels)
       Te_loss = Te_loss + loss_2.item()
       test_loss.append(Te_loss)

       _, predicted = torch.max(test_Output, 1) 
       correct_test += (predicted == labels).sum().item()
       accuracy_test = (correct_test / len(test_dataset)) * 100
       print(f"test_loss : {Te_loss},Accuracy for test : {accuracy_test}%")
     
   

 
            


            
 
            
