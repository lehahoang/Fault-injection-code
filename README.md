This repository contains the Python source code for project "DNN Fault injection". It was developed on top of Pytorch framework.
The main functions of the simulator:
- Training Deep Neural Network (DNN) models
- Doing image classification on the test set
- Injecting bit flips at memory blocks storing DNNs' parameters
- Evaluating the resilience of the network in faulty environment

Output of the fault injection simulation:
- Accuracy curve over the defined range of fault rates in .csv file 
- Summary of the stats regarding the locations which are injected, e.g, e7 or sign bit of the number.
- A '.png' file showing the accuracy curve (optinal)


Dataset used in the project:
- MNIST
- CIFAR-10
- CIFAR-100

DNN models used in the project:
- Lenet-5
- AlexNet
- GoogleNet
- VGG-16 (comming soon)
- MobileNet (comming soon)

I declare that I do not own the copyright of the source code of implementing DNN models. They were taken from other github repositories.  

Data representation used in the project:
- Single-precision
- (Un)signed Integer (comming soon)

Fault models:
- Random bit-flips
- Stuck-at-faults (coming soon)

Recommended fault ranges for experiments:
- Comming soon

Quick tutorial to use this simulator:
- Coming soon
