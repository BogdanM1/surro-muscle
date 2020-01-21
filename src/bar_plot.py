import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ['G1', 'G2']
orig_means = [20, 34]
surro_means = [25, 32]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, orig_means, width, label='Original model', color='rebeccapurple')
rects2 = ax.bar(x + width/2, surro_means, width, label='Surrogate model', color='lightcoral')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('ylabel')
ax.set_title('Title')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper left', frameon=False)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.ylim(bottom=0, top=1.2*plt.ylim()[1])
plt.savefig('bar_plot.png')