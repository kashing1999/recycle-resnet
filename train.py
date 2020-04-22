import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math
from model import convNet

plt.ion()

data_dir = 'data'
CHECK_POINT_PATH = 'checkpoint.tar'

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([224,224]),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x),
                                           data_transforms[x]) for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Classes:")
for c in class_names:
    print("    " + c)
print(f'Train image size: {dataset_sizes["train"]}')
print(f'Validation image size: {dataset_sizes["val"]}')

def imshow(inp, title=None):
    """Imshow for Tensor"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if title != None:
        plt.title(title)
    plt.imshow(inp)
    plt.pause(5)

#inputs, classes = next(iter(dataloaders['train']))

#sample_train_images = torchvision.utils.make_grid(inputs)

#imshow(sample_train_images, title=classes)

def train_model(model, criterion, optimizer, scheduler, num_epochs=2, checkpoint=None):
    since = time.time()

    best_less = 0

    if checkpoint is None:
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = math.inf
        best_acc = 0.
    else:
        print(f'Val loss: {checkpoint["best_val_loss"]}, Val accuracy: {checkpoint["best_val_accuracy"]}')
        model.load_state_dict(checkpoint['model_state_dict'])
        best_model_wts = copy.deepcopy(model.state_dict())
        optimizer.load_state_dict(checkpoint['model_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_loss = checkpoint['best_val_loss']
        best_acc = checkpoint['best_val_accuracy']

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if i % 200 == 199:
                    print('[%d, %d] loss: %.3f' % (epoch + 1, i, running_loss / (i * inputs.size(0))))

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if phase == 'train':
                    scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_loss < best_loss:
                if epoch != 0:
                    print(f'New best model found!')
                    print(f'New record loss: {epoch_loss:4f}, previous_record loss: {best_loss:4f}')
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f} Best val loss: {:.4f}'.format(best_acc, best_loss))

    model.load_state_dict(best_model_wts)
    return model, best_loss, best_acc

model_conv = convNet(num_classes)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# parameters = [{'params': model_conv.conv.fc.parameters()}, {'params': model_conv.fc1.parameters()}, {'params': model_conv.fc2.parameters()}]

optimizer_conv = optim.SGD(model_conv.conv.fc.parameters, lr=0.1, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=6, gamma=0.5)

try:
    checkpoint = torch.load(CHECK_POINT_PATH)
    print("checkpoint loaded")
except:
    checkpoint = None
    print("checkpoint not found")

model_conv, best_val_loss, best_val_acc = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=3, checkpoint=checkpoint)

torch.save(model_conv.state_dict(), './weights.pth')

print("Loss: ", end="")
print(best_val_loss)
print("Acc: ", end="")
print(best_val_acc)
