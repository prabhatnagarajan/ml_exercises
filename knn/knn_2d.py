# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs, make_moons, make_circles

# def generate_datasets(
#     n_blobs=3,
#     n_samples=300,
#     cluster_std=2.5,       # higher spread → more overlap
#     noise_level=0.5,       # adds small Gaussian noise to all blob points
#     include_nonlinear=True,
#     random_state=42
# ):
#     """
#     Generate 2D datasets for k-NN experimentation:
#       - Overlapping Gaussian blobs
#       - Optional nonlinear 'moons' and 'circles'
#     """
#     np.random.seed(random_state)

#     # --- 1️⃣ Overlapping Gaussian blobs ---
#     Xb, yb = make_blobs(
#         n_samples=n_samples,
#         n_features=2,
#         centers=n_blobs,
#         cluster_std=cluster_std,
#         random_state=random_state
#     )

#     # Add extra Gaussian noise to make blobs less dense
#     Xb += np.random.normal(scale=noise_level, size=Xb.shape)

#     datasets = [(Xb, yb, f"{n_blobs} overlapping blobs (σ={cluster_std})")]

#     # --- 2️⃣ Optional nonlinear datasets ---
#     if include_nonlinear:
#         Xm, ym = make_moons(n_samples=n_samples, noise=0.15, random_state=random_state)
#         Xc, yc = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=random_state)
#         datasets += [
#             (Xm, ym, "Moons (nonlinear)"),
#             (Xc, yc, "Circles (nonlinear)")
#         ]

#     # --- 3️⃣ Plot all datasets ---
#     ncols = len(datasets)
#     fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

#     if ncols == 1:
#         axes = [axes]  # make iterable for consistency

#     for ax, (X, y, title) in zip(axes, datasets):
#         ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=40, edgecolor='k')
#         ax.set_title(title)
#         ax.grid(True)

#     plt.tight_layout()
#     plt.show()

#     return datasets

# # --- Example usage ---
# datasets = generate_datasets(
#     n_blobs=5,         # number of Gaussian clusters
#     n_samples=200,     # total samples per dataset
#     cluster_std=2.5,   # higher → more overlap
#     noise_level=0.5,   # add global noise for fuzziness
#     include_nonlinear=True
# )

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.neighbors import KNeighborsClassifier

def experiment(k, data_type="blobs"):
    """
    Experiment with k-NN classifier on 2D datasets.
    Visualize decision boundaries for different datasets.
    """
    if data_type == "blobs":
        X, y = make_blobs(
        n_samples=300,
        centers=3,
        n_features=2,
        cluster_std=2.0,
        random_state=42
            )
    elif data_type == "moons":
        X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
    elif data_type == "circles":
        X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)

# --- Step 2: Fit k-NN classifier ---
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    # --- Step 3: Create a meshgrid covering the feature space ---
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.05),
        np.arange(y_min, y_max, 0.05)
    )

    # --- Step 4: Predict class for each point in the grid ---
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # --- Step 5: Plot decision boundary and training points ---
    plt.figure(figsize=(7, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')  # decision regions
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, edgecolor='k')
    plt.title(f"k-NN Decision Boundary (k={k})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    experiment(k=5, data_type="blobs")    # Experiment on overlapping blobs
    experiment(k=5, data_type="moons")    # Experiment on moons
    experiment(k=5, data_type="circles")  # Experiment on circles