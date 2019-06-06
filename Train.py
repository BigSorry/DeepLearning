import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import VGG as net
import random

def run():

    cfg = {
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    }

    model = net.VGG(net.VGG.make_layers(cfg['A'], True), 10)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Training
    bestErrorRate = 1
    deterioration = 0
    batches = len(iter(testloader))
    validationSize = int(batches*0.5)
    while deterioration < 5:
        runningLoss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            #runningLoss += loss.item()

    #         if i % 2000 == 1999:  # print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.3f' %
    #                   (epoch + 1, i + 1, runningLoss / 2000))
    #             runningLoss = 0.0

        # Testing phase
        correct = 0
        total = 0
        validationIndices = set(random.sample(range(batches), validationSize))
        batchIndex = -1
        with torch.no_grad():
            for data in testloader:
                batchIndex+=1
                if batchIndex in validationIndices:
                    images, labels = data
                    if torch.cuda.is_available():
                        images = images.cuda()
                        labels = labels.cuda()
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        errorRate = 1 - (correct / total)
        if bestErrorRate > errorRate:
            bestErrorRate = errorRate
            torch.save(model.state_dict(), "models/model")
            print("save with error rate {}".format(errorRate))
            deterioration = 0
        else:
            deterioration += 1

if __name__ == "__main__":
    run()


