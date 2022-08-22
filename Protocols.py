import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import Nets


class Net:
    
    def __init__(self):
        self._layers    = []
        self._variables = []
        self._body      = None
        self._inference = None
        self._loss      = None
        self._updateOp  = None

        self._graph = tf.Graph()
        self._sess = tf.compat.v1.Session(graph=self._graph)

        with self._graph.as_default():
            self._ifTest        = tf.Variable(False, name='ifTest', trainable=False, dtype=tf.bool)
            self._step          = tf.Variable(0, name='step', trainable=False, dtype=tf.int32)
            self._phaseTrain    = tf.compat.v1.assign(self._ifTest, False)
            self._phaseTest     = tf.compat.v1.assign(self._ifTest, True)

        self.generator_loss_history = []
        self.generator_accuracy_history = []
        self.simulator_loss_history = []
        self.simulator_accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []

    # define body of either target model, or of simulator if subclass is adversarial model
    def body(self, images, architecture, num_middle=2, for_generator=False):
        # preprocess images
        standardized = self.preproc(images)

        # when we duplicate the simulator, with the same weights, and tie it to the generator output, we do not want to
        # double the regularisation losses for each layer. Therefore, we pass in an empty list, so that layers are not
        # added a second time to self._layers. If subclass is a target model, this argument should always be False
        if for_generator:
            layers = []
        else:
            layers = self._layers

        if architecture == "SimpleNet":
            net = Nets.SimpleNet(standardized, self._step, self._ifTest, layers)
        elif architecture == "SmallNet":
            net = Nets.SmallNet(standardized, self._step, self._ifTest, layers)
        elif architecture == "ConcatNet":
            net = Nets.ConcatNet(standardized, self._step, self._ifTest, layers)
        elif architecture == "Xception":
            net = Nets.Xception(standardized, self._step, self._ifTest, layers, numMiddle=num_middle)
        else:
            raise ValueError("Invalid simulator architecture argument!")

        return net.output

    def preproc(self, images):
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

    def plot_training_history(self, model, test_after):
        plt.plot(self.simulator_loss_history, label="Simulator")
        plt.plot(self.generator_loss_history, label="Generator")
        test_steps = list(range(0, len(self.simulator_loss_history) + 1, test_after))
        plt.plot(test_steps, self.test_loss_history, label="Generator Test")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("{} loss history".format(model))
        plt.legend()
        plt.show()

        plt.plot(self.simulator_accuracy_history, label="Simulator")
        plt.plot(self.generator_accuracy_history, label="Generator")
        test_steps = list(range(0, len(self.simulator_accuracy_history) + 1, test_after))
        plt.plot(test_steps, self.test_accuracy_history, label="Generator Test")
        plt.xlabel("Steps")
        plt.ylabel("TFR")
        plt.title("{} TFR history".format(model))
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
    
