import numpy as np
import matplotlib.pyplot as plt
lengths = np.array([350, 325, 300, 275, 250, 225, 175, 150, 125, 100, 50])
accuracy_500 = np.array([[60.51,67.77,69.44],
[61.00,72.60,76.86],
[62.49,77.50,85.18],
[61.19,83.58,85.23],
[60.08,83.13,84.63],
[60.63,81.46,83.90],
[62.17,75.06,78.30],
[61.86,70.32,74.34],
[63.81,68.05,71.69],
[65.58,66.74,70.33],
[64.75,71.99,71.00]])

def plot_length(accuracy, lengths):
    range_min_max = np.array([accuracy.min()-3, accuracy.max() + 3])

    plt.plot(lengths, accuracy[:,0], 'o-', label='orig')
    plt.plot(lengths, accuracy[:,1], 's-', label='drop')
    plt.plot(lengths, accuracy[:,2], 'x-',label='beta_drop')
    plt.plot(np.array([300, 300]), range_min_max, label='Test length=Train length')
    plt.xlabel("Length of sequences")
    plt.ylabel("Accuracy (%)")
    plt.ylim([accuracy.min()-3,  accuracy.max() + 3])
    plt.legend()
    plt.show()

def plot_length_error(error, lengths):
    range_min_max = np.array([error.min()-3, error.max() + 3])

    plt.plot(lengths, error[:,0], 'o-', label='orig')
    plt.plot(lengths, error[:,1], 's-', label='drop')
    plt.plot(lengths, error[:,2], 'x-',label='beta_drop')
    plt.plot(np.array([300, 300]), range_min_max, label='Test length=Train length')
    plt.xlabel("Length of sequences")
    plt.ylabel("Error (%)")
    plt.ylim([error.min()-3,  error.max() + 3])
    plt.legend()
    plt.show()

accuracy_20000 = np.array([[71.05,68.9,67.42],
[79.64,75.81,77.45],
[97.97,87.1,99.91],
[97.43,99.72,99.92],
[96.37,99.7,99.88],
[94.01,99.34,99.85],
[89.01,95.52,99.74],
[87.68,83.67,96.56],
[84.09,90.54,93.71],
[77.44,80.32,84.29],
[64.92,69.71,72.07]])

accuracy_2000=np.array([[68.51,68.46,69.01],
[73.95,73.94,77.67],
[79.14,83.65,96.85],
[78.05,96.29,96.7],
[75.76,96.21,96.58],
[74.53,95.37,95.31],
[69.7,84.45,74.36],
[67.54,74.9,63.43],
[67.94,65.63,57.5],
[60.59,57.71,52.51],
[60.31,53.11,50.96]])
print(len(lengths))
print(accuracy_500.shape)
print(accuracy_20000.shape)

# plot_length(accuracy=accuracy_2000, lengths=lengths)
plot_length_error(error=100 - accuracy_500, lengths=lengths)