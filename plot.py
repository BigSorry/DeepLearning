import pickle
import matplotlib.pyplot as plt


path = "D:/Semester2/Deep learning/DeepLearning/info_dict.pickle"
with open(path, 'rb') as file:
    data = pickle.load(file)

for image in data.keys():
    if len(data[image]) > 0:
        plt.figure()
        for model, info in data[image].items():
            x = info[:, 0]
            y = info[:, 1]
            plt.plot(x, y)
        plt.legend(data[image].keys())

plt.show()
