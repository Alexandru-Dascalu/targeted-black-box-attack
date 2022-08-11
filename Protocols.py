import numpy as np
import os
import matplotlib.pyplot as plt

class Net:
    
    def __init__(self):
        self._layers    = []
        self._variables = []
        self._body      = None
        self._inference = None
        self._loss      = None
        self._updataOp  = None

        self.generator_loss_history = []
        self.generator_accuracy_history = []
        self.simulator_loss_history = []
        self.simulator_accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []
        
    def body(self):
        pass
    
    def inference(self):
        pass
    
    def loss(self):
        pass
    
    def train(self):
        pass
    
    def evaluate(self):
        pass

    def load_training_history(self, path):
        assert type(path) is str

        if os.path.exists(path):
            array_dict = np.load(path)

            self.simulator_loss_history = array_dict['arr_0']
            self.simulator_accuracy_history = array_dict['arr_1']
            self.generator_loss_history = array_dict['arr_2']
            self.generator_accuracy_history = array_dict['arr_3']
            self.test_loss_history = array_dict['arr_4']
            self.test_accuracy_history = array_dict['arr_5']
            print("Training history restored.")

    def plot_training_history(self, model):
        plt.plot(self.simulator_loss_history, label="Simulator")
        plt.plot(self.generator_loss_history, label="Generator")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("{} loss history".format(model))
        plt.legend()
        plt.show()

        plt.plot(self.test_loss_history, label="Test")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("{} test loss history".format(model))
        plt.legend()
        plt.show()

        plt.plot(self.simulator_accuracy_history, label="Simulator")
        plt.plot(self.generator_accuracy_history, label="Generator")
        plt.xlabel("Steps")
        plt.ylabel("TFR")
        plt.title("{} TFR history".format(model))
        plt.legend()
        plt.show()

        plt.plot(self.test_accuracy_history, label="Test")
        plt.xlabel("Steps")
        plt.ylabel("TFR")
        plt.title("{} test TFR history".format(model))
        plt.legend()
        plt.show()
    
    @property
    def summary(self): 
        summs = []
        summs.append("=>Network Summary: ")
        for elem in self._layers:
            summs.append(elem.summary)
        summs.append("<=Network Summary: ")
        return "\n\n".join(summs)
    
