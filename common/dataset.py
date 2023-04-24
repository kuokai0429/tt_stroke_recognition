# 2023.0420.0253 @Brian

class StrokeRecognitionDataset:

    def __init__(self, features, targets):

        self.features = features
        self.targets = targets
        self.classes = {"其他": 0, "正手發球": 1, "反手發球": 2, "正手推球": 3, "反手推球": 4, "正手切球": 5, "反手切球":6}
        
    def __len__(self):

        return self.features.shape[0]
    
    def __getitem__(self, idx):

        return self.features[idx], self.targets[idx]