from dataclasses import dataclass

@dataclass
class CommonArgs:
    dataset: str
    split: str
    data_path: str
    data_url: str

@dataclass
class DataloaderArgs:
    seed: int
    radius: int
    n_bits: int
    val_frac: float
    test_frac: float
    batch_size: int

@dataclass
class MLPRegArgs:
    num_layers: int
    num_nodes_per_layer: int
    learning_rate: float
    epochs: int

@dataclass
class PyTorchArgs:
    common_args: CommonArgs
    dataloader_args: DataloaderArgs
    mlpreg_args: MLPRegArgs
