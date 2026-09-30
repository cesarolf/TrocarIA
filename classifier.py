"""Rede inicial para futura classificação supervisionada, ainda sem rótulos clínicos."""

from torch import nn

from analysis import ERROR_CATEGORIES


CLASS_LABELS = ERROR_CATEGORIES
INPUT_FEATURES = len(CLASS_LABELS)
HIDDEN_UNITS = (64, 32)
DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
VALIDATION_FRACTION = 0.2


class WritingClassifier(nn.Module):
    def __init__(self, number_of_classes=len(CLASS_LABELS)):
        super().__init__()
        if number_of_classes < 2:
            raise ValueError("Defina ao menos duas classes antes de treinar o classificador.")
        self.layers = nn.Sequential(
            nn.Linear(INPUT_FEATURES, HIDDEN_UNITS[0]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_UNITS[0], HIDDEN_UNITS[1]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_UNITS[1], number_of_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, features):
        return self.layers(features)