import pickle
import matplotlib.pyplot as plt


path = "info_dict"
with open(path, 'rb') as file:
    data = pickle.load(file)

for image in data.keys():
    if len(data[image]) > 0:
        fig = plt.figure()
        for model, info in data[image].items():
            # info[0] contains loss info and info[1] contains duration in seconds
            lossInfo = info[0]
            x = lossInfo[:, 0]
            y = lossInfo[:, 1]
            plt.plot(x, y)
        plt.legend(data[image].keys())
        plt.xlabel('Optimization iterations')
        plt.ylabel('Total loss')
        fig.savefig('plots/{}'.format(image))


