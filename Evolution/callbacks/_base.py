# BaseCallback
from typing import List

class BaseCallback:
    """trainer를 바로 참조함
    
    Keras 참고
    """
    def on_train_begin(self, trainer) -> None:
        ...
    
    def on_train_end(self, trainer) -> None:
        ...
    
    def on_step_begin(self, trainer) -> None:
        ...
    
    def on_step_end(self, trainer) -> None:
        ...


class CallbackManager:
    
    def __init__(
        self,
        callbacks: List[BaseCallback],
    ):
        self.callbacks = callbacks
    
    def on_train_begin(self, trainer):
        [cb.on_train_begin(trainer) for cb in self.callbacks]
        
    def on_train_end(self, trainer):
        [cb.on_train_end(trainer) for cb in self.callbacks]
        
    def on_step_begin(self, trainer):
        [cb.on_step_begin(trainer) for cb in self.callbacks]
        
    def on_step_end(self, trainer):
        [cb.on_step_end(trainer) for cb in self.callbacks]
        