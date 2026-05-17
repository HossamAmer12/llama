import torch

class KMeans:
    def __init__(self, k: int, max_iters: int = 100):
        self.k         = k
        self.max_iters = max_iters
        self.centroids = None   # shape: [k, d]

    def fit(self, X: torch.Tensor):
        # X: [n, d]
        # 1. randomly initialize centroids from X
        # 2. loop for max_iters:
        #       a. assign each point to nearest centroid
        #       b. recompute centroids as mean of assigned points
        #       c. check convergence (centroids stopped moving)
        pass

    def _assign(self, X: torch.Tensor) -> torch.Tensor:
        # X: [n, d],  centroids: [k, d]
        # return labels: [n]  (index of nearest centroid per point)
        pass

    def _update(self, X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # return new centroids: [k, d]
        # hint: mean of all points assigned to each cluster
        pass

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        # return labels: [n]
        return self._assign(X)


# --- toy data ---
torch.manual_seed(0)
X = torch.cat([
    torch.randn(100, 2) + torch.tensor([0., 0.]),   # cluster A
    torch.randn(100, 2) + torch.tensor([5., 5.]),   # cluster B
    torch.randn(100, 2) + torch.tensor([0., 5.]),   # cluster C
], dim=0)   # shape: [300, 2]

km = KMeans(k=3)
km.fit(X)
labels = km.predict(X)
print(labels)