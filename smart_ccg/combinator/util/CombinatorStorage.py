class CombinatorStorage:
    def __init__(self, storage_root):
        self.storage_root = storage_root
        self.dataset_path = storage_root / "dataset"

    def save_dataset(self, examples):
        pass
