import numpy as np
import matplotlib.pyplot as plt
import math

##====================================================================
length_acc_1000 = np.array([[315,66.21,71.95,77.69,72.64,78.75],
[310,66.95,74.74,80.87,75.63,82.49],
[305,66.97,76.70,85.00,78.18,86.42],
[300,68.23,78.46,88.91,80.00,90.69],
[295,67.50,81.28,89.15,82.30,90.93],
[290,67.94,82.86,89.18,84.58,90.97],
[285,68.06,85.74,89.33,87.18,91.24],
[280,68.09,87.99,89.82,89.21,91.29],
[275,67.66,88.32,89.29,88.94,91.02],
[270,67.01,88.28,88.98,88.99,90.61],
[265,67.77,88.82,89.74,89.71,90.89],
[260,67.45,88.24,89.42,88.88,90.55],
[255,66.82,88.08,89.12,89.09,90.41],
[250,66.82,87.81,88.83,88.32,89.72],
[225,65.96,86.08,87.43,86.50,87.52],
[200,65.95,82.29,83.20,81.91,82.96]])
# lengths = np.array([350, 325, 315, 310, 305, 300, 295, 290, 285, 280,
#                     275, 250, 225, 200, 175, 150, 125, 100, 50])
#
# accuracy_500 = np.array([[0.5832,0.6502,0.6845],
# [0.5945,0.6924,0.7502],
# [0.5934,0.7000,0.7726],
# [0.6013,0.7149,0.7947],
# [0.5950,0.7341,0.8199],
# [0.6053,0.7444,0.8377],
# [0.5963,0.7616,0.8408],
# [0.5999,0.7755,0.8461],
# [0.6028,0.8031,0.8501],
# [0.6056,0.8210,0.8477],
# [0.6007,0.8230,0.8464],
# [0.6032,0.8192,0.8403],
# [0.5976,0.8092,0.8339],
# [0.6091,0.7713,0.8143],
# [0.6097,0.7305,0.7696],
# [0.6144,0.6862,0.7317],
# [0.6244,0.6437,0.6892],
# [0.6367,0.6521,0.6881],
# [0.6593,0.6555,0.6826]])
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
    pos_offset = 2.5
    n = len(lengths)
    x = np.arange(n, 0, -1)
    print(x)
    idx = np.where(lengths == 300)[0][0]
    print(idx)
    idx = x[idx]
    line_width = 2.5
    range_min_max = np.array([error.min()-pos_offset, error.max() + pos_offset])
    plt.plot(x, error[:,0], 'o-', label='Baseline (Clean Data)', linewidth=line_width)
    plt.plot(x, error[:,1], 's-', label='SpanDrop', linewidth=line_width)
    plt.plot(x, error[:,2], 'x-', label='Beta-SpanDrop', linewidth=line_width)
    plt.plot(x, error[:, 3], 'v-', label='SpanDrop-Oracle', linewidth=line_width)
    plt.plot(x, error[:, 4], '<-', label='Beta-SpanDrop-Oracle', linewidth=line_width)
    plt.plot(np.array([idx, idx]), range_min_max, 'k--', linewidth=line_width, label='Test seq length=Train seq length')
    plt.xlabel("Length of sequences in test stage")
    plt.ylabel("Error (%)")
    plt.xticks(ticks=x, labels=lengths)
    plt.ylim([error.min()-pos_offset,  error.max() + pos_offset])
    plt.legend(loc=6, bbox_to_anchor=(0.1, 0.6))
    plt.savefig('error_vs_length.pdf', dpi=300)
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
lengths = length_acc_1000[:,0].astype(int)
accuracy_1000 = length_acc_1000[:,1:6]
error = 100 - accuracy_1000
# plot_length_error(error=error, lengths=lengths)
##====================================================================
drop_vs_mask = np.array([68.23,78.46,88.91,72.88,73.55])
def drop_vs_mask_plot(drop_vs_mask_data):
    pos_offset = 2.5
    barWidth = 0.005
    # set height of bar
    clean_data = [drop_vs_mask_data[0], drop_vs_mask_data[0]]
    SpanDrop_data = [drop_vs_mask_data[1], drop_vs_mask_data[2]]
    SpanMask_data = [drop_vs_mask_data[3], drop_vs_mask_data[4]]

    # Set position of bar on X axis
    br1 = np.array([0.5, 0.55])
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, clean_data, color='r', width=barWidth,
            edgecolor='grey', label='Clean Data')
    plt.bar(br2, SpanDrop_data, color='g', width=barWidth,
            edgecolor='grey', label='SpanDrop')
    plt.bar(br3, SpanMask_data, color='b', width=barWidth,
            edgecolor='grey', label='SpanMask')

    # Adding Xticks
    # plt.xlabel('Distribution')
    plt.ylabel('Error (%)')
    plt.xticks([r + barWidth for r in br1],
               ['Bernoulli', 'Beta-Bernoulli'])
    plt.ylim([drop_vs_mask_data.min() - pos_offset, drop_vs_mask_data.max() + 2* pos_offset])
    plt.xlim([0.49, 0.57])
    plt.legend(loc=9, mode='expand', ncol=3)
    plt.savefig('drop_vs_mask.pdf', dpi=300)

    plt.show()
drop_vs_mask_plot(100 - drop_vs_mask)
##====================================================================
fixed_position = np.array([50.16,64.47,68.77])
topk_position = np.array([67.39,67.94,74.69])
zero_shot = np.array([52.71,61.21,63.91])

def pos_zeroshot_plot(fixed_pos_data, topk_pos_data, zero_shot_data):
    pos_offset = 2.5
    data = np.stack([fixed_pos_data, topk_pos_data, zero_shot_data])
    barWidth = 0.05
    # set height of bar
    clean_data = [fixed_pos_data[0], topk_pos_data[0], zero_shot_data[0]]
    SpanDrop_data = [fixed_pos_data[1], topk_pos_data[1], zero_shot_data[1]]
    BetaSpanDrop_data = [fixed_pos_data[2], topk_pos_data[2], zero_shot_data[2]]

    # Set position of bar on X axis
    br1 = np.arange(len(clean_data)) * 1.0 / 3
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, clean_data, color='r', width=barWidth,
            edgecolor='grey', label='Clean Data')
    plt.bar(br2, SpanDrop_data, color='g', width=barWidth,
            edgecolor='grey', label='SpanDrop')
    plt.bar(br3, BetaSpanDrop_data, color='m', width=barWidth,
            edgecolor='grey', label='BetaSpanDrop')

    # Adding Xticks
    plt.ylabel('Error (%)')
    plt.xticks([r + barWidth for r in br1],
               ['Fixed position', 'Position in Top 100', 'Zero-Shot'])
    plt.ylim([data.min() - pos_offset, data.max() + 1.25 * pos_offset])
    plt.legend(loc=9, mode='expand', ncol=3)
    plt.savefig('position_zero_shot.pdf', dpi=300)

    plt.show()

# pos_zeroshot_plot(fixed_pos_data=100 - fixed_position, topk_pos_data=100 -  topk_position, zero_shot_data= 100 -zero_shot)