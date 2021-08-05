# Data file
import torch
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


def main():
    x, y, z = get_dls_MNIST(32)

    x, y = next(iter(x))
    print(x.shape)
    print(y.shape)

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

    def __init__(self, X = None, Y = None, k = 3, p = 2):
        self.k = k
        super().__init__(X, Y, p)

    def __call__(self, x):
        return self.predict(x)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1/self.p)

        knn = dist.topk(self.k, largest=False)
        votes = self.train_label[knn.indices]

        winner = torch.zeros(votes.size(0), dtype=votes.dtype, device=votes.device)
        count = torch.zeros(votes.size(0), dtype=votes.dtype, device=votes.device) - 1

        for lab in self.unique_labels:
            vote_count = (votes == lab).sum(1)
            who = vote_count >= count
            winner[who] = lab
            count[who] = vote_count[who]

        return winner

if __name__ == "__main__":
    main()
