import pickle as pk

import matplotlib.pyplot as plt

with open('../results/torch_mlp.pkl', 'rb') as f:
    data = pk.load(f)

losses_train = data["train_loss"]
losses_test = data["test_loss"]
acc_train = data["train_acc"]
acc_test = data["test_acc"]
x = [i for i in range(len(acc_train))]
plt.title("Accuracy MLP torch")
plt.plot(x, acc_test, label="acc_test")
plt.plot(x, acc_train, label="acc_train")
plt.legend()
plt.savefig("../results/acc_mlp_torch.png")
plt.show()
