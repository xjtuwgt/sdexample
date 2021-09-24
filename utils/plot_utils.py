import numpy as np
import matplotlib.pyplot as plt
import math
lengths = np.array([350, 325, 315, 310, 305, 300, 295, 290, 285, 280,
                    275, 250, 225, 200, 175, 150, 125, 100, 50])

accuracy_500 = np.array([[0.5832,0.6502,0.6845],
[0.5945,0.6924,0.7502],
[0.5934,0.7000,0.7726],
[0.6013,0.7149,0.7947],
[0.5950,0.7341,0.8199],
[0.6053,0.7444,0.8377],
[0.5963,0.7616,0.8408],
[0.5999,0.7755,0.8461],
[0.6028,0.8031,0.8501],
[0.6056,0.8210,0.8477],
[0.6007,0.8230,0.8464],
[0.6032,0.8192,0.8403],
[0.5976,0.8092,0.8339],
[0.6091,0.7713,0.8143],
[0.6097,0.7305,0.7696],
[0.6144,0.6862,0.7317],
[0.6244,0.6437,0.6892],
[0.6367,0.6521,0.6881],
[0.6593,0.6555,0.6826]])
# accuracy_500 = np.array([[60.51,67.77,69.44],
# [61.00,72.60,76.86],
# [62.49,77.50,85.18],
# [61.19,83.58,85.23],
# [60.08,83.13,84.63],
# [60.63,81.46,83.90],
# [62.17,75.06,78.30],
# [61.86,70.32,74.34],
# [63.81,68.05,71.69],
# [65.58,66.74,70.33],
# [64.75,71.99,71.00]])

def plot_length(accuracy, lengths):
    pos_offset = 1
    range_min_max = np.array([accuracy.min()-pos_offset, accuracy.max() + pos_offset])
    n = len(lengths)
    x = np.arange(n, 0, -1)
    print(x)
    idx = np.where(lengths == 300)[0][0]
    print(idx)
    idx = x[idx]
    plt.plot(x, accuracy[:,0], 'o-', label='Baseline')
    plt.plot(x, accuracy[:,1], 's-', label='SpanDrop')
    plt.plot(x, accuracy[:,2], 'x-',label='Beta-SpanDrop')
    plt.plot(np.array([idx, idx]), range_min_max, label='Test length = Train length')
    plt.xlabel("Length of sequences")
    plt.ylabel("Accuracy (%)")
    plt.xticks(ticks=x, labels=lengths)
    plt.ylim([accuracy.min()-pos_offset,  accuracy.max() + pos_offset])
    plt.legend()
    plt.show()

def plot_length_error(error, lengths):
    pos_offset = 1
    n = len(lengths)
    x = np.arange(n, 0, -1)
    print(x)
    idx = np.where(lengths == 300)[0][0]
    print(idx)
    idx = x[idx]
    line_width = 2.5
    range_min_max = np.array([error.min()-pos_offset, error.max() + pos_offset])
    plt.plot(x, error[:,0], 'o-', label='Baseline', linewidth=line_width)
    plt.plot(x, error[:,1], 's-', label='SpanDrop', linewidth=line_width)
    plt.plot(x, error[:,2], 'x-', label='Beta-SpanDrop', linewidth=line_width)
    plt.plot(np.array([idx, idx]), range_min_max, '--', linewidth=line_width, label='Test length=Train length')
    plt.xlabel("Length of sequences in test stage")
    plt.ylabel("Error (%)")
    plt.xticks(ticks=x, labels=lengths)
    plt.ylim([error.min()-pos_offset,  error.max() + pos_offset])
    plt.legend()
    plt.savefig('eror_vs_length.pdf', dpi=300)
    plt.show()


# accuracy_20000 = np.array([[71.05,68.9,67.42],
# [79.64,75.81,77.45],
# [97.97,87.1,99.91],
# [97.43,99.72,99.92],
# [96.37,99.7,99.88],
# [94.01,99.34,99.85],
# [89.01,95.52,99.74],
# [87.68,83.67,96.56],
# [84.09,90.54,93.71],
# [77.44,80.32,84.29],
# [64.92,69.71,72.07]])
#
# accuracy_2000=np.array([[68.51,68.46,69.01],
# [73.95,73.94,77.67],
# [79.14,83.65,96.85],
# [78.05,96.29,96.7],
# [75.76,96.21,96.58],
# [74.53,95.37,95.31],
# [69.7,84.45,74.36],
# [67.54,74.9,63.43],
# [67.94,65.63,57.5],
# [60.59,57.71,52.51],
# [60.31,53.11,50.96]])
# print(len(lengths))
# print(accuracy_500.shape)
# print(accuracy_20000.shape)
#
# plot_length(accuracy=accuracy_500[:-5] * 100, lengths=lengths[:-5])
error = 100 - 100 * accuracy_500
plot_length_error(error=error[:-5][3:], lengths=lengths[:-5][3:])