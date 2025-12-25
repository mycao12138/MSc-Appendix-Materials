import os
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class DGraphFinDataset(InMemoryDataset):
    """
    DGraphFin dataset loader with basic financial feature processing.

    Pipeline:
    1) Load raw data from dgraphfin.npz.
    2) Impute missing values (mean for continuous, mode for categorical).
    3) Apply log1p to continuous features (with per-column shift if negatives exist).
    4) Standardize features using Z-score.
    5) Encode temporal edge attribute as normalized relative time: (t_max - t_edge) / span.
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["dgraphfin.npz"]

    @property
    def processed_file_names(self):
        return ["dgraphfin_processed.pt"]

    def download(self):
        raw_path = os.path.join(self.raw_dir, "dgraphfin.npz")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                "Please download 'dgraphfin.npz' manually and place it under "
                f"'{self.raw_dir}/'."
            )

    def process(self):
        print("Processing DGraphFin dataset...")

        raw_path = os.path.join(self.raw_dir, "dgraphfin.npz")
        data_np = np.load(raw_path, allow_pickle=True)

        x = data_np["x"]  # (N, F)
        y = data_np["y"]  # (N,)
        edge_index = data_np["edge_index"]  # (2, M)
        edge_timestamp = data_np["edge_timestamp"]  # (M,)

        train_mask = data_np["train_mask"] if "train_mask" in data_np else None
        valid_mask = data_np["valid_mask"] if "valid_mask" in data_np else (
            data_np["val_mask"] if "val_mask" in data_np else None
        )
        test_mask = data_np["test_mask"] if "test_mask" in data_np else None

        # NOTE: Adjust these indices according to the official DGraphFin schema.
        continuous_cols = list(range(10))
        categorical_cols = list(range(10, 17))

        if x.ndim != 2:
            raise ValueError(f"Expected x to be 2D, but got shape {x.shape}.")

        print(f"Original feature shape: {x.shape}")

        # 1) Missing value imputation
        if np.isnan(x).any():
            print("Imputing missing values...")

            if len(continuous_cols) > 0:
                imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
                x[:, continuous_cols] = imp_mean.fit_transform(x[:, continuous_cols])

            if len(categorical_cols) > 0:
                imp_mode = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
                x[:, categorical_cols] = imp_mode.fit_transform(x[:, categorical_cols])

        # 2) Log-transform continuous features (log1p with per-column shift if needed)
        if len(continuous_cols) > 0:
            print("Applying log1p transformation to continuous features...")
            min_val = np.min(x[:, continuous_cols], axis=0)
            shift = np.maximum(-min_val, 0.0)  # shift each column to be non-negative
            x[:, continuous_cols] = np.log1p(x[:, continuous_cols] + shift)

        # 3) Standardization (Z-score) across all features
        print("Applying Z-score standardization...")
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()

        # 4) Temporal edge attribute: normalized relative time
        print("Processing temporal edge attributes...")
        edge_index = torch.from_numpy(edge_index).long()
        edge_time = torch.from_numpy(edge_timestamp).float()

        t_max = edge_time.max()
        edge_attr_time = t_max - edge_time  # larger means older

        denom = edge_attr_time.max()
        eps = 1e-12
        edge_attr_time = edge_attr_time / (denom + eps)

        edge_attr = edge_attr_time.unsqueeze(1)  # (M, 1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        if train_mask is not None:
            data.train_mask = torch.from_numpy(train_mask).bool()
        if valid_mask is not None:
            data.valid_mask = torch.from_numpy(valid_mask).bool()
        if test_mask is not None:
            data.test_mask = torch.from_numpy(test_mask).bool()

        # Labeled mask for supervised learning (adjust if dataset uses other label conventions)
        data.labeled_mask = (data.y == 0) | (data.y == 1)

        print("Data processing complete.")
        print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")

        torch.save(self.collate([data]), self.processed_paths[0])


if __name__ == "__main__":
    root_dir = "./data"

    try:
        dataset = DGraphFinDataset(root=root_dir)
        data = dataset[0]

        print("\n=== Dataset Statistics ===")
        print(f"X shape: {data.x.shape}")
        print(f"Edge index shape: {data.edge_index.shape}")
        print(f"Edge attr shape: {data.edge_attr.shape}")
        print(f"Unique labels: {torch.unique(data.y)}")

        print(f"Feature mean (first 5): {data.x.mean(dim=0)[:5]}")
        print(f"Feature std  (first 5): {data.x.std(dim=0)[:5]}")

        print(f"Max relative time: {data.edge_attr.max()}")
        print(f"Min relative time: {data.edge_attr.min()}")

        if hasattr(data, "train_mask"):
            print(f"Train labeled nodes: {int(data.train_mask.sum())}")
        if hasattr(data, "valid_mask"):
            print(f"Valid labeled nodes: {int(data.valid_mask.sum())}")
        if hasattr(data, "test_mask"):
            print(f"Test labeled nodes: {int(data.test_mask.sum())}")

    except Exception as e:
        print(f"\n[Error] {e}")
        print("Please ensure 'dgraphfin.npz' is present under './data/raw/'.")