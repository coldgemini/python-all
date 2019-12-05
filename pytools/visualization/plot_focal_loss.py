import numpy as np
import matplotlib.pyplot as plt

gamma_list = [0, 0.5, 1, 2, 3, 4, 5, 6, 7]
# gamma = np.array(gamma)
pred = np.linspace(0.1, 1, 100)

gamma = 2


def fc_loss(pred, gamma):
    loss = -np.power((1 - pred), gamma) * np.log(pred)
    return loss


# plt.hist(count_list, bins='auto')
# plt.hist(count_list, bins=np.arange(0, 40000, 100))
# plt.hist(count_list, bins=np.arange(0, 200, 5))
# plt.show()

# fig = plt.figure()
# ax = plt.axes()
#
# ax.plot(pred, loss)
plt.xlim([0.8, 1])
plt.ylim([0, 0.0001])
for gamma in gamma_list:
    loss = fc_loss(pred, gamma)
    plt.plot(pred, loss, label=f"gamma={gamma}")

plt.xlabel('pred')
plt.ylabel('loss')
plt.title('focal loss')
plt.legend()
# plt.axis('tight')
# plt.axis('on')
plt.tight_layout()
plt.grid(True)

plt.show()
