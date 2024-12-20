from enum import Enum

class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class PoseDataset(Dataset):