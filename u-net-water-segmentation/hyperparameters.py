import torch


class Hyperparameters:
    NUM_CLASSES = 2
    EPOCHS = 10
    TRAIN_VAL_SPLIT = 0.8
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    HEIGHT = 256    # in the current implementation height and width both need to be divisable by 16
    WIDTH  = 256
    STARTING_EPOCH = 0
    LR = 0.001
    DEVICE = "cuda" if torch.cuda.is_available() else "mps"



