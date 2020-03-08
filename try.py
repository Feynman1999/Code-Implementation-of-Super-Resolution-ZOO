import matplotlib.pyplot as plt
import numpy as np
from util import util
import os

plt.plot([1,2], [1,2])
plt.show()

# leg = ["test1", "test2", "test3"]
# X = np.linspace(0, 1000, 1)
# Y = np.random.normal(size=(1, 3))
# moving_average = [10, 100]
# for ma in moving_average:
#     ma_Y = util.moving_average(Y, ma=ma)
#     X = np.linspace(X[0], X[-1], ma_Y.shape[0])
#     gap = int(np.ceil(ma_Y.shape[0] / 10000))
#     # save svg
#     title = "loss from epoch {:.2f} to epoch {:.2f}  moving_average: {}".format(X[0], X[-1], ma)
#     plt.figure(figsize=(len(X[::gap]) / 100 + 1, 8))
#     print(len(X[::gap]))
#     for i in range(ma_Y.shape[1]):
#         plt.plot(X[::gap], ma_Y[:, i][::gap], label=leg[i])  # promise that <= 10000
#     plt.title(title)
#     plt.xlabel('epochs')
#     plt.ylabel('Losses')
#     plt.legend()
#     fig = plt.gcf()
#     fig.savefig(os.path.join(".", str(ma) + '.svg'), dpi=600, bbox_inches='tight')
#     # plt.show()
#     plt.clf()

