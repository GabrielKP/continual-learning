# Data file
import torch
import math
from torch.utils.data import Dataset, DataLoader
from mnist import MNIST

# MNIST
class DatasetMNIST(Dataset):

    def __init__(self, X, y, device):
        assert len(X) == len(y), "Not same amount of labels and data"
        self.X = (torch.tensor(X) / 255).to(device)
        self.y = torch.tensor(y).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dls_MNIST(batch_size, device, split_val=0.92, small=False):
    mndata = MNIST(path="./data", gz=True)

    X_train_raw, y_train_raw = mndata.load_training()
    X_test_raw, y_test_raw = mndata.load_testing()

    if small:
        small_split = 5000
        X_train_raw = X_train_raw[:small_split]
        y_train_raw = y_train_raw[:small_split]
        small_split = 1000
        X_test_raw = X_test_raw[:small_split]
        y_test_raw = y_test_raw[:small_split]

    idx_split = int(len(X_train_raw) * split_val)

    train_ds = DatasetMNIST(
        X_train_raw[:idx_split], y_train_raw[:idx_split], device)
    valid_ds = DatasetMNIST(
        X_train_raw[idx_split:], y_train_raw[idx_split:], device)
    test_ds = DatasetMNIST(X_test_raw, y_test_raw, device)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, valid_dl, test_dl

# kNN

def distance_matrix(x, y=None, p = 2): #pairwise distance of vectors

    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)

    return dist

# from https://gist.github.com/JosueCom/7e89afc7f30761022d7747a501260fe3
class KNN():

    def __init__(self, device, X = None, y = None, k = 3, p = 2):
        self.k = k
        self.p = p
        self.X = None
        self.y = None
        self.unique_labels = torch.tensor([0,]).to(device)
        self.device = device
        if X is not None:
            self.train(X, y)

    def __call__(self, x):
        return self.predict(x)

    def train(self, X, y):
        X = X.to(self.device)
        y = y.to(self.device)
        if self.X is None:
            size, data_size = X.shape
            self.idx = 0
            self.X = torch.empty((1024, data_size), dtype=X.dtype)
            self.y = torch.empty((1024,), dtype=y.dtype)

        # check if there is enough space
        while self.y.shape[0] - self.idx <= y.shape[0]:
            self.doubleXy()

        batch_size = y.shape[0]
        self.X[self.idx:self.idx + batch_size] = X
        self.y[self.idx:self.idx + batch_size] = y
        self.idx += batch_size

        if (not torch.equal(self.unique_labels, y.unique(sorted=True))):
            self.unique_labels = self.y.unique(sorted=True).to(self.device)
            self.k = int(16 * math.log(self.unique_labels.shape[0], 2))

    def doubleXy(self):
        size, data_size = self.X.shape
        self.X.resize_((size * 2, data_size))
        self.y.resize_((size * 2))

    def predict(self, x):
        if type(self.X) == type(None) or type(self.y) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        x = x.to(self.device)

        dist = distance_matrix(x, self.X[:self.idx], self.p) ** (1/self.p)

        knn = dist.topk(self.k, largest=False)
        votes = self.y[knn.indices]

        winner = torch.zeros(votes.size(0), dtype=votes.dtype, device=votes.device)
        count = torch.zeros(votes.size(0), dtype=votes.dtype, device=votes.device) - 1

        for lab in self.unique_labels:
            vote_count = (votes == lab).sum(1)
            who = vote_count >= count
            winner[who] = lab
            count[who] = vote_count[who]

        return winner


def main():
    knn = KNN("cpu", k=3,p=200)
    x, y, z = get_dls_MNIST(64, "cpu", small=True)

    import time

    print(len(x))

    start_time = time.time()
    for _ in range(10):
        for X, y in x:
            knn.train(X, y)
    end_time = time.time()
    passed = end_time - start_time
    print(passed)
    return
    t = iter(x)
    X, y = next(t)
    X2, y2 = next(t)
    X1, y1 = next(t)
    knn.train(X, y)
    print("1")
    knn.train(X, y)
    print("2")
    print(X[0], X2[0])
    knn.train(X2, y2)
    res = knn.predict(X)
    print(res)

if __name__ == "__main__":
    main()
