"""
K-Means Clustering — algorithm steps
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Goal: partition N points into K clusters so that each point
      belongs to the cluster with the nearest centroid.

Step 1 — Initialise centroids
         Pick K points at random from the dataset as the starting centroids.

Step 2 — Assign each point to the nearest centroid
         For every data point, compute its distance to all K centroids.
         Label it with the index of the closest one.

Step 3 — Recompute centroids
         For each cluster, take the mean of all points assigned to it.
         That mean becomes the new centroid.
         If a cluster is empty, keep the previous centroid unchanged.

Step 4 — Check for convergence
         If the centroids did not move (or moved less than a tiny threshold),
         the algorithm has converged — stop.
         Otherwise go back to Step 2.

Step 5 — Predict
         To label new points, run Step 2 with the final centroids.
"""

import torch
import matplotlib.pyplot as plt

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
        
        # Step 1: Initialize centroids randomly from X
        n, d = X.shape
        indices = torch.randperm(n)[:self.k]
        self.centroids = X[indices]
        
        # Step 2: Iteratively update centroids
        for _ in range(self.max_iters):
            # a. Assign each point to nearest centroid
            labels = self._assign(X)
            
            # b. Recompute centroids as mean of assigned points
            new_centroids = self._update(X, labels)
            
            # c. Check for convergence (if centroids do not change)
            if torch.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
        
        
    def _assign(self, X: torch.Tensor) -> torch.Tensor:
        # X: [n, d],  centroids: [k, d]
        # return labels: [n]  (index of nearest centroid per point)
        
        # Compute L2 distances from points to centroids
        distances = torch.cdist(X, self.centroids, p=2)  # shape: [n, k]
        return torch.argmin(distances, dim=1)  # shape: [n]

    def _update(self, X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # return new centroids: [k, d]
        # hint: mean of all points assigned to each cluster
        new_centroids = []
        for i in range(self.k):
            mask = labels == i
            if mask.sum() > 0:
                new_centroids.append(X[mask].mean(dim=0))
            else:
                new_centroids.append(self.centroids[i])
        return torch.stack(new_centroids)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        # return labels: [n]
        return self._assign(X)


def plot_clusters(X: torch.Tensor, labels, centroids):
    X_np = X.numpy()
    colors = ['#7F77DD', '#1D9E75', '#D85A30']

    plt.figure(figsize=(6, 5))

    if labels is None:
        plt.scatter(X_np[:, 0], X_np[:, 1], c='gray', alpha=0.5, s=20)
    else:
        labels_np = labels.numpy()
        for i in range(len(colors)):
            mask = labels_np == i
            plt.scatter(X_np[mask, 0], X_np[mask, 1],
                        c=colors[i], alpha=0.5, s=20, label=f'cluster {i}')
        plt.legend()

    if centroids is not None:
        centroids_np = centroids.numpy()
        plt.scatter(centroids_np[:, 0], centroids_np[:, 1],
                    c='black', marker='X', s=200, zorder=5, label='centroids')

    plt.title('KMeans clustering')
    plt.tight_layout()
    plt.savefig('kmeans.png', dpi=150)
    plt.show()

# --- toy data ---
torch.manual_seed(0)
X = torch.cat([
    torch.randn(100, 2) + torch.tensor([0., 0.]),   # cluster A
    torch.randn(100, 2) + torch.tensor([5., 5.]),   # cluster B
    torch.randn(100, 2) + torch.tensor([0., 5.]),   # cluster C
], dim=0)   # shape: [300, 2]
print("Data shape:", X.shape)

km = KMeans(k=3)
km.fit(X)
labels = km.predict(X)
print(labels)

plot_clusters(X, labels, km.centroids)


