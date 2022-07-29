import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import Layers
import Nets
import MNIST
import Preproc

wd = 1e-4
NoiseRange = 255.0


def create_generator(images, targets, num_experts, step, ifTest, layers):
    # define encoder as a CNN with 4 conv2d layers
    with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE) as scope:
        encoder = Layers.Conv2D(Preproc.normalise_images(images), convChannels=16, convKernel=[5, 5], convStride=[2, 2],
                                conv_weight_decay=wd, convInit=Layers.XavierInit, convPadding='SAME',
                                biasInit=Layers.ConstInit(0.0), batch_normalisation=True, step=step, ifTest=ifTest,
                                epsilon=1e-5,
                                activation=Layers.ReLU, name='Conv1', dtype=tf.float32)
        layers.append(encoder)
        encoder = Layers.Conv2D(encoder.output, convChannels=32, convKernel=[5, 5], convStride=[2, 2],
                                conv_weight_decay=wd, convInit=Layers.XavierInit, convPadding='SAME',
                                biasInit=Layers.ConstInit(0.0), batch_normalisation=True, step=step, ifTest=ifTest,
                                epsilon=1e-5,
                                activation=Layers.ReLU, name='Conv2', dtype=tf.float32)
        layers.append(encoder)
        encoder = Layers.Conv2D(encoder.output, convChannels=64,
                                convKernel=[3, 3], convStride=[2, 2], conv_weight_decay=wd,
                                convInit=Layers.XavierInit, convPadding='SAME',
                                biasInit=Layers.ConstInit(0.0),
                                batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                activation=Layers.ReLU,
                                name='Conv7b', dtype=tf.float32)
        layers.append(encoder)
        encoder = Layers.DeConv2D(encoder.output, convChannels=64, shapeOutput=[7, 7],
                                  convKernel=[3, 3], convStride=[2, 2], conv_weight_decay=wd,
                                  convInit=Layers.XavierInit, convPadding='SAME',
                                  biasInit=Layers.ConstInit(0.0),
                                  batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                  activation=Layers.ReLU,
                                  reuse=tf.AUTO_REUSE, name='G_DeConv192', dtype=tf.float32)
        layers.append(encoder)

    # define decoder
    with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE) as scope:
        subnets_output = []
        for idx in range(num_experts):
            subnet = Layers.DeConv2D(encoder.output, convChannels=32, shapeOutput=[14, 14],
                                     convKernel=[5, 5], convStride=[2, 2], conv_weight_decay=wd,
                                     convInit=Layers.XavierInit, convPadding='SAME',
                                     biasInit=Layers.ConstInit(0.0),
                                     batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                     activation=Layers.ReLU,
                                     reuse=tf.AUTO_REUSE, name='G_DeConv96_' + str(idx), dtype=tf.float32)
            layers.append(subnet)
            subnet = Layers.DeConv2D(subnet.output, convChannels=16, shapeOutput=[28, 28],
                                     convKernel=[5, 5], convStride=[2, 2], conv_weight_decay=wd,
                                     convInit=Layers.XavierInit, convPadding='SAME',
                                     biasInit=Layers.ConstInit(0.0),
                                     batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                     activation=Layers.ReLU,
                                     reuse=tf.AUTO_REUSE, name='G_DeConv48_' + str(idx), dtype=tf.float32)
            layers.append(subnet)
            # why is this a convolution layer and not deconv?
            subnet = Layers.Conv2D(subnet.output, convChannels=1,
                                   convKernel=[3, 3], convStride=[1, 1], conv_weight_decay=wd,
                                   convInit=Layers.NormalInit(0.01), convPadding='SAME',
                                   biasInit=Layers.ConstInit(0.0),
                                   batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                   activation=Layers.Linear,
                                   reuse=tf.AUTO_REUSE, name='G_SepConv3_' + str(idx), dtype=tf.float32)
            layers.append(subnet)
            subnets_output.append(tf.expand_dims(subnet.output, axis=-1))

        subnets_output = tf.concat(subnets_output, axis=-1)
        expert_weights_layer = Layers.FullyConnected(tf.one_hot(targets, 10), outputSize=num_experts,
                                                     weightInit=Layers.XavierInit, wd=wd,
                                                     biasInit=Layers.ConstInit(0.0),
                                                     activation=Layers.Softmax,
                                                     reuse=tf.AUTO_REUSE, name='G_WeightsMoE', dtype=tf.float32)
        layers.append(expert_weights_layer)

        # re-shapes subnets from batch_size x 28 x 28 x 1 x num_experts to 28 x 28 x 1 x batch_size x num_experts, so
        # that it can element-wise multiplied with the expert weights, which have a shape of batch_size x num_experts
        subnets_output = tf.transpose(subnets_output, [1, 2, 3, 0, 4])

        # performs element wise multiplication to apply the weights for each subnet output, then re-shape back into the
        # original shape: batch_size x 28 x 28 x 1 x num_experts
        mixture_of_experts = tf.transpose(subnets_output * expert_weights_layer.output,
                                          [3, 0, 1, 2, 4])
        # output of decoder is an average of the weighted outputs of each expert
        mixture_of_experts = tf.reduce_mean(mixture_of_experts, -1)

        # apply weighted sum layer activation
        noises = tf.nn.tanh(mixture_of_experts) * NoiseRange
        print('Shape of Noises: ', noises.shape)

    encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
    decoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')

    return noises, encoder_variables, decoder_variables


def create_simulator(images, step, ifTest, layers):
    # define simulator with an architecture almost identical to SimpleNet in the paper
    net = Layers.DepthwiseConv2D(Preproc.normalise_images(tf.clip_by_value(images, 0, 255)), convChannels=16,
                                 convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                                 convInit=Layers.XavierInit, convPadding='SAME',
                                 biasInit=Layers.ConstInit(0.0),
                                 bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=Layers.ReLU,
                                 name='P_DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=32,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='P_SepConv96', dtype=tf.float32)
    layers.append(net)

    toadd = Layers.Conv2D(net.output, convChannels=64,
                          convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='P_SepConv192Shortcut', dtype=tf.float32)
    layers.append(toadd)

    net = Layers.SepConv2D(net.output, convChannels=64,
                           convKernel=[3, 3], convStride=[2, 2], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='P_SepConv192a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=64,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           name='P_SepConv192b', dtype=tf.float32)
    layers.append(net)

    added = toadd.output + net.output

    toadd = Layers.Conv2D(added, convChannels=128,
                          convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='P_SepConv384Shortcut', dtype=tf.float32)
    layers.append(toadd)

    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU384')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=128,
                           convKernel=[3, 3], convStride=[2, 2], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='P_SepConv384a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=128,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='P_SepConv384b', dtype=tf.float32)
    layers.append(net)

    added = toadd.output + net.output

    toadd = Layers.Conv2D(added, convChannels=256,
                          convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='P_SepConv768Shortcut', dtype=tf.float32)
    layers.append(toadd)

    net = Layers.Activation(added, activation=Layers.ReLU, name='P_ReLU768')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=256,
                           convKernel=[3, 3], convStride=[2, 2], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='P_SepConv768a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=256,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='P_SepConv768b', dtype=tf.float32)
    layers.append(net)

    added = toadd.output + net.output

    net = Layers.Activation(added, activation=Layers.ReLU, name='P_ReLU11024')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=512,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='P_SepConv1024', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='P_GlobalAvgPool')
    layers.append(net)
    logits = Layers.FullyConnected(net.output, outputSize=10, weightInit=Layers.XavierInit, wd=wd,
                                   biasInit=Layers.ConstInit(0.0),
                                   activation=Layers.Linear,
                                   reuse=tf.AUTO_REUSE, name='P_FC_classes', dtype=tf.float32)
    layers.append(logits)

    return logits.output


# why does this not add its layers to the layers list?
def create_simulator_G(images, step, ifTest):
    # create identical structure to the net in create_predictor, including the same layer names
    net = Layers.DepthwiseConv2D(Preproc.normalise_images(tf.clip_by_value(images, 0, 255)), convChannels=16,
                                 convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                                 convInit=Layers.XavierInit, convPadding='SAME',
                                 biasInit=Layers.ConstInit(0.0),
                                 bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=Layers.ReLU,
                                 name='P_DepthwiseConv3x16', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=32,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='P_SepConv96', dtype=tf.float32)

    toadd = Layers.Conv2D(net.output, convChannels=64,
                          convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='P_SepConv192Shortcut', dtype=tf.float32)

    net = Layers.SepConv2D(net.output, convChannels=64,
                           convKernel=[3, 3], convStride=[2, 2], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='P_SepConv192a', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=64,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           name='P_SepConv192b', dtype=tf.float32)

    added = toadd.output + net.output

    toadd = Layers.Conv2D(added, convChannels=128,
                          convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='P_SepConv384Shortcut', dtype=tf.float32)

    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU384')
    net = Layers.SepConv2D(net.output, convChannels=128,
                           convKernel=[3, 3], convStride=[2, 2], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='P_SepConv384a', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=128,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='P_SepConv384b', dtype=tf.float32)

    added = toadd.output + net.output

    toadd = Layers.Conv2D(added, convChannels=256,
                          convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='P_SepConv768Shortcut', dtype=tf.float32)

    net = Layers.Activation(added, activation=Layers.ReLU, name='P_ReLU768')
    net = Layers.SepConv2D(net.output, convChannels=256,
                           convKernel=[3, 3], convStride=[2, 2], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='P_SepConv768a', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=256,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='P_SepConv768b', dtype=tf.float32)

    added = toadd.output + net.output

    net = Layers.Activation(added, activation=Layers.ReLU, name='P_ReLU11024')
    net = Layers.SepConv2D(net.output, convChannels=512,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='P_SepConv1024', dtype=tf.float32)
    net = Layers.GlobalAvgPool(net.output, name='P_GlobalAvgPool')
    logits = Layers.FullyConnected(net.output, outputSize=10, weightInit=Layers.XavierInit, wd=wd,
                                   biasInit=Layers.ConstInit(0.0),
                                   activation=Layers.Linear,
                                   reuse=tf.AUTO_REUSE, name='P_FC_classes', dtype=tf.float32)

    return logits.output


HParamMNIST = {'BatchSize': 200,
               'NumSubnets': 10,
               'NumPredictor': 1,
               'NumGenerator': 1,
               'NoiseDecay': 1e-5,
               'LearningRate': 1e-3,
               'MinLearningRate': 1e-5,
               'DecayAfter': 300,
               'ValidateAfter': 300,
               'TestSteps': 50,
               'TotalSteps': 60000}


class AdvNetMNIST(Nets.Net):

    def __init__(self, image_shape, enemy, hyper_params=None):
        Nets.Net.__init__(self)

        if hyper_params is None:
            hyper_params = HParamMNIST

        self.hyper_params = hyper_params
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)

        # the targeted neural network model
        self._enemy = enemy

        with self._graph.as_default():
            # variable to keep check if network is being tested or trained
            self._ifTest = tf.Variable(False, name='ifTest', trainable=False, dtype=tf.bool)
            # define operations to set ifTest variable
            self._phaseTrain = tf.assign(self._ifTest, False)
            self._phaseTest = tf.assign(self._ifTest, True)

            self._step = tf.Variable(0, name='step', trainable=False, dtype=tf.int32)

            # Inputs
            self._images = tf.placeholder(dtype=tf.float32, shape=[self.hyper_params['BatchSize']] + image_shape,
                                          name='MNIST_images')
            self._labels = tf.placeholder(dtype=tf.int64, shape=[self.hyper_params['BatchSize']],
                                          name='MNIST_labels')
            self._adversarial_targets = tf.placeholder(dtype=tf.int64, shape=[self.hyper_params['BatchSize']],
                                                       name='MNIST_targets')

            # define generator
            with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE) as scope:
                self._generator, self._varsGE, self._varsGD = create_generator(self._images, self._adversarial_targets,
                                                                               self.hyper_params['NumSubnets'],
                                                                               self._step,
                                                                               self._ifTest, self._layers)
            self._noises = self._generator
            self._adversarial_images = self._noises + self._images

            # define simulator
            with tf.variable_scope('Predictor', reuse=tf.AUTO_REUSE) as scope:
                self._predictor = create_simulator(self._images, self._step, self._ifTest, self._layers)

                # what is the point of this??? Why is the generator training against a different simulator, which is
                # not trained to match the target model? Why is one simulator trained on normal images, and another on
                # adversarial images?
                self._predictorG = create_simulator_G(self._adversarial_images, self._step, self._ifTest)

            # define inference as hard label prediction of simulator on natural images
            self._inference = self.inference(self._predictor)
            # accuracy is how often simulator prediction matches the prediction of the target net
            self._accuracy = tf.reduce_mean(tf.cast(tf.equal(self._inference, self._labels), tf.float32))

            self._loss = 0
            for elem in self._layers:
                if len(elem.losses) > 0:
                    for tmp in elem.losses:
                        self._loss += tmp

            self._updateOps = []
            for elem in self._layers:
                if len(elem.update_ops) > 0:
                    for tmp in elem.update_ops:
                        self._updateOps.append(tmp)

            # simulator loss matches simulator output against output of target model
            self._lossSimulator = self.loss(self._predictor, self._labels, name='lossP') + self._loss
            # generator trains to produce perturbations that make the simulator produce the desired target labels
            self._lossGenerator = self.loss(self._predictorG, self._adversarial_targets, name='lossG') + \
                                  self.hyper_params['NoiseDecay'] * tf.reduce_mean(tf.norm(self._noises)) + self._loss
            print(self.summary)
            print("\n Begin Training: \n")

            # Saver
            self._saver = tf.train.Saver(max_to_keep=5)

    def inference(self, logits):
        return tf.argmax(logits, axis=-1, name='inference')

    def loss(self, logits, labels, name='cross_entropy'):
        net = Layers.CrossEntropy(logits, labels, name=name)
        self._layers.append(net)
        return net.output

    def train(self, training_data_generator, test_data_generator, path_load=None, path_save=None):
        with self._graph.as_default():
            # perhaps this should use an exponential decay rate instead?
            self._lr = tf.Variable(self.hyper_params['LearningRate'], trainable=False)
            self._lrDecay1 = tf.assign(self._lr, self._lr * 0.1)

            self._stepInc = tf.assign(self._step, self._step + 1)

            self._varsG = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
            self._varsP = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Predictor')

            # define optimiser without the minimisation operation. This is done later, after the gradients are clipped
            self._optimizerG = tf.train.AdamOptimizer(self._lr, epsilon=1e-8)
            gradientsG = self._optimizerG.compute_gradients(self._lossGenerator, var_list=self._varsG)
            # clip gradients
            clipped_gradients = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradientsG]
            self._optimizerG = self._optimizerG.apply_gradients(clipped_gradients)

            # define simulator optimisation operation
            self._optimizerS = tf.train.AdamOptimizer(self._lr, epsilon=1e-8).minimize(self._lossSimulator,
                                                                                       var_list=self._varsP)

            # Initialize all variables
            self._sess.run(tf.global_variables_initializer())

            # check if it should re-start training from a known checkpoint
            if path_load is not None:
                self.load(path_load)
            else:
                # Warm up simulator by training it to match target model hard labels with normal images as input
                print('Warming up. ')
                for idx in range(300):
                    data, _, _ = next(training_data_generator)
                    # get hard label prediction of target model
                    target_model_labels = np.array(self._enemy.infer(data))
                    loss, tfr, _ = self._sess.run([self._lossSimulator, self._accuracy, self._optimizerS],
                                                  feed_dict={self._images: data, self._labels: target_model_labels})

                    print('\rPredictor => Step: ', idx - 300,
                          '; Loss: %.3f' % loss,
                          '; Accuracy: %.3f' % tfr,
                          end='')

                # test pre-trained model on some test set data
                warmupAccu = 0.0
                for idx in range(50):
                    data, _, _ = next(test_data_generator)
                    target_model_labels = np.array(self._enemy.infer(data))
                    loss, tfr, _ = self._sess.run([self._lossSimulator, self._accuracy, self._optimizerS],
                                                  feed_dict={self._images: data, self._labels: target_model_labels})
                    warmupAccu += tfr / 50
                print('\nWarmup Test Accuracy: ', warmupAccu)

            self.evaluate(test_data_generator)

            self._sess.run([self._phaseTrain])
            if path_save is not None:
                self.save(path_save)

            # Start main training loop
            globalStep = 0
            # self._sess.run(self._lrDecay2)
            while globalStep < self.hyper_params['TotalSteps']:
                self._sess.run(self._stepInc)

                # train simulator for a couple of steps
                for _ in range(self.hyper_params['NumPredictor']):
                    # ground truth labels are not needed for training the simulator
                    data, _, target_label = next(training_data_generator)
                    # adds Random uniform noise to normal data
                    data = data + (np.random.rand(self.hyper_params['BatchSize'], 28, 28, 1) - 0.5) * NoiseRange * 2

                    # perform one optimisation step to train simulator so it has the same predictions as the target
                    # model does on normal images with noise
                    target_model_labels = self._enemy.infer(data)
                    loss, tfr, globalStep, _ = self._sess.run([self._lossSimulator, self._accuracy, self._step,
                                                               self._optimizerS],
                                                              feed_dict={self._images: data,
                                                                         self._labels: target_model_labels})

                    # generate adversarial image
                    adversarial_images = self._sess.run(self._adversarial_images,
                                                        feed_dict={self._images: data,
                                                                   self._adversarial_targets: target_label})
                    # perform one optimisation step to train simulator so it has the same predictions as the target
                    # model does on adversarial images
                    target_model_labels = self._enemy.infer(adversarial_images)
                    loss, tfr, globalStep, _ = self._sess.run([self._lossSimulator, self._accuracy, self._step,
                                                               self._optimizerS],
                                                              feed_dict={self._images: adversarial_images,
                                                                         self._labels: target_model_labels})

                    print('\rPredictor => Step: ', globalStep,
                          '; Loss: %.3f' % loss,
                          '; Accuracy: %.3f' % tfr,
                          end='')

                # train generator for a couple of steps
                for _ in range(self.hyper_params['NumGenerator']):
                    data, _, target_label = next(training_data_generator)
                    target_model_labels = self._enemy.infer(data)

                    # check that target labels are different than what the target model already predicts
                    for idx in range(data.shape[0]):
                        if target_model_labels[idx] == target_label[idx]:
                            tmp = random.randint(0, 9)
                            while tmp == target_model_labels[idx]:
                                tmp = random.randint(0, 9)
                            target_label[idx] = tmp

                    loss, adversarial_images, globalStep, _ = self._sess.run([self._lossGenerator,
                                                                              self._adversarial_images, self._step,
                                                                              self._optimizerG],
                                                                             feed_dict={self._images: data,
                                                                                        self._adversarial_targets: target_label})
                    target_model_adversarial_predictions = self._enemy.infer(adversarial_images)
                    tfr = np.mean(target_label == target_model_adversarial_predictions)
                    ufr = np.mean(target_model_labels != target_model_adversarial_predictions)

                    print('\rGenerator => Step: ', globalStep,
                          '; Loss: %.3f' % loss,
                          '; TFR: %.3f' % tfr,
                          '; UFR: %.3f' % ufr,
                          end='')

                # evaluate on test set data every once in a while
                if globalStep % self.hyper_params['ValidateAfter'] == 0:
                    self.evaluate(test_data_generator)

                    # this code is just for generating some more adversarial noise for test images, and printing
                    # information about them
                    data, _, target_label = next(test_data_generator)
                    adversarial_images = self._sess.run(self._adversarial_images,
                                                        feed_dict={self._images: data,
                                                                   self._adversarial_targets: target_label})

                    target_model_labels = self._enemy.infer(data)
                    target_model_adversarial_predictions = self._enemy.infer(adversarial_images)
                    print((adversarial_images - data)[1, 10:15, 10:15])
                    print((adversarial_images - data).max())
                    print((adversarial_images - data).min())
                    print(list(zip(target_model_labels, target_model_adversarial_predictions, target_label)))

                    if path_save is not None:
                        self.save(path_save)
                    self._sess.run([self._phaseTrain])

                # if globalStep == 1501:
                # self._sess.run(self._lrDecay1)

                if (globalStep % 5400 == 0 or globalStep % 8100 == 0) and globalStep < 10000:
                    self._sess.run(self._lrDecay1)
                    print('Learning rate decayed. ')

    def evaluate(self, test_data_generator, path=None):
        if path is not None:
            self.load(path)

        total_loss = 0.0
        total_ufr = 0.0
        total_tfr = 0.0

        self._sess.run([self._phaseTest])
        for _ in range(self.hyper_params['TestSteps']):
            data, _, target_labels = next(test_data_generator)
            target_model_labels = self._enemy.infer(data)

            # for each batch image, make sure target label is different than the predicted label by the target model
            for idx in range(data.shape[0]):
                if target_model_labels[idx] == target_labels[idx]:
                    tmp = random.randint(0, 9)
                    while tmp == target_model_labels[idx]:
                        tmp = random.randint(0, 9)
                    target_labels[idx] = tmp

            # evaluate generator loss on test data
            loss, adversarial_images = self._sess.run([self._lossGenerator, self._adversarial_images],
                                                      feed_dict={self._images: data,
                                                                 self._adversarial_targets: target_labels})

            adversarial_images = adversarial_images.clip(0, 255).astype(np.uint8)
            target_model_adversarial_predictions = self._enemy.infer(adversarial_images)

            tfr = np.mean(target_labels == target_model_adversarial_predictions)
            # ufr should really be based on ground truth label, not what the target model predicted on the normal image
            ufr = np.mean(target_model_labels != target_model_adversarial_predictions)

            total_loss += loss
            total_tfr += tfr
            total_ufr += ufr

        total_loss /= self.hyper_params['TestSteps']
        total_tfr /= self.hyper_params['TestSteps']
        total_ufr /= self.hyper_params['TestSteps']
        print('\nTest: Loss: ', total_loss,
              '; TFR: ', total_tfr,
              '; UFR: ', total_ufr)

    def sample(self, test_data_generator, path=None):
        if path is not None:
            self.load(path)

        self._sess.run([self._phaseTest])
        data, label, target = next(test_data_generator)
        target_model_labels = self._enemy.infer(data)

        # for each test image, check that the adversarial target is different than what the target model already
        # predicts on the normal image
        for idx in range(data.shape[0]):
            if target_model_labels[idx] == target[idx]:
                tmp = random.randint(0, 9)
                while tmp == target_model_labels[idx]:
                    tmp = random.randint(0, 9)
                target[idx] = tmp

        # create adversarial images
        loss, adversarial_images = self._sess.run([self._lossGenerator, self._adversarial_images],
                                                  feed_dict={self._images: data,
                                                             self._adversarial_targets: target})
        adversarial_images = adversarial_images.clip(0, 255).astype(np.uint8)
        adversarial_labels = self._enemy.infer(adversarial_images)

        for idx in range(10):
            for jdx in range(3):
                # show sampled adversarial image
                plt.subplot(10, 6, idx * 6 + jdx * 2 + 1)
                plt.imshow(data[idx * 3 + jdx, :, :, 0], cmap='gray')
                plt.subplot(10, 6, idx * 6 + jdx * 2 + 2)
                plt.imshow(adversarial_images[idx * 3 + jdx, :, :, 0], cmap='gray')
                # print target model prediction on original image, the prediction on adversarial image, and target label
                print([target_model_labels[idx * 3 + jdx], adversarial_labels[idx * 3 + jdx], target[idx * 3 + jdx]])
        plt.show()

    def save(self, path):
        self._saver.save(self._sess, path, global_step=self._step)

    def load(self, path):
        self._saver.restore(self._sess, path)


if __name__ == '__main__':
    enemy = MNIST.NetMNIST([28, 28, 1], 2)
    enemy.load('./ClassifyMNIST/netmnist.ckpt-39900')

    net = AdvNetMNIST([28, 28, 1], enemy=enemy)
    batchTrain, batchTest = MNIST.get_adversarial_data_generators(batch_size=HParamMNIST['BatchSize'],
                                                                  preproc_size=[28, 28, 1])
    net.train(batchTrain, batchTest, path_save='./AttackMNIST/netmnist.ckpt')  # 90 and 94, 87 and 89
    while True:
        net.sample(batchTest, './AttackMNIST/netmnist.ckpt-4800')

    # NN: Accu:  0.8760999999999997 ; FullRate:  0.8996999999999999
    # 255: Accu:  0.9229999999999997 ; FullRate:  0.9318999999999997
