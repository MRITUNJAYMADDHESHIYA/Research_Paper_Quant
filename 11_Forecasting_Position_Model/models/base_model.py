from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass

