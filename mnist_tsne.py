import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import datasets
from sklearn import manifold


# Loading data
data = datasets.fetch_openml(
    name='mnist_784',
    version=1,
    return_X_y=True)

pixel_values, targets = data
targets = targets.astype(int)

# Reshaping one pixel_values file to have a sample image
single_image = pixel_values[1, :].reshape((28, 28))

tsne = manifold.TSNE(
    n_components=2,
    random_state=42)

transformed_data = tsne.fit_transform(
    X=pixel_values[:3000, :])

tsne_df = pd.DataFrame(
    np.column_stack((transformed_data, targets[:3000])),
    columns=["x", "y", "targets"])

tsne_df.loc[:, "targets"] = tsne_df.targets.astype(int)

if __name__ == "__main__":
    plt.imshow(single_image, cmap="gray")

    grid = sns.FacetGrid(tsne_df, hue="targets", height=8)

    grid.map(plt.scatter, "x", "y").add_legend()

    plt.show()
