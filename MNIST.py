import random
import h5py
import numpy as np

import tensorflow as tf

import Preproc
import Layers
import Nets


def load_HDF5():
    """Loads MNIST data set from .h5 file and returns it as a tuple of Numpy arrays.

    Returns
    -------
    dataTrain
        Numpy array with the training images.
    labelsTrain
        Numpy array with the training labels.
    dataTest
        Numpy array with the test imgaes.
    labelsTest
        Numpy array with the test labels.
    """
    with h5py.File('MNIST.h5', 'r') as f:
        dataTrain = np.expand_dims(np.array(f['Train']['images'])[:, :, :, 0], axis=-1)
        labelsTrain = np.array(f['Train']['labels']).reshape([-1])
        dataTest = np.expand_dims(np.array(f['Test']['images'])[:, :, :, 0], axis=-1)
        labelsTest = np.array(f['Test']['labels']).reshape([-1])

    return dataTrain, labelsTrain, dataTest, labelsTest


def generate_data_label_pair(data, labels):
    index = Preproc.generate_index(data.shape[0], shuffle=True)
    while True:
        indexAnchor = next(index)
        imageAnchor = data[indexAnchor]
        labelAnchor = labels[indexAnchor]

        image = [imageAnchor]
        label = [labelAnchor]

        yield image, label


def get_data_generators(batch_size, preproc_size=None):
    """ Creates generators that generate batches of samples from the MNIST data set.
    Parameters
    ----------
    batch_size : int
        Size of the batches generated by the generators returned by this method.
    preproc_size : list of ints
        Shape of each sample after pre-processing.

    Returns
    ----------
    Two generators: the first is for generating training data set batches, and the second is for generating test set
    batches.
    """
    if preproc_size is None:
        preproc_size = [28, 28, 1]
    (dataTrain, labelsTrain, dataTest, labelsTest) = load_HDF5()

    # preprocess training and test images by making sure they are 28x28x1
    def preprocess_images(images, size):
        results = np.ndarray([images.shape[0]] + size, np.uint8)
        for idx in range(images.shape[0]):
            distorted = images[idx]
            results[idx] = distorted.reshape([28, 28, 1])

        return results

    def generate_batch(training):
        if training:
            generator = generate_data_label_pair(dataTrain, labelsTrain)
        else:
            generator = generate_data_label_pair(dataTest, labelsTest)
        while True:
            batchImages = []
            batchLabels = []
            for _ in range(batch_size):
                images, labels = next(generator)
                batchImages.append(images)
                batchLabels.append(labels)
            batchImages = preprocess_images(np.concatenate(batchImages, axis=0), preproc_size)
            batchLabels = np.concatenate(batchLabels, axis=0)

            yield batchImages, batchLabels

    return generate_batch(training=True), generate_batch(training=False)


def get_adversarial_data_generators(batch_size, preproc_size=None):
    """ Creates generators that generate batches of samples from the MNIST data set, with pre-processed specifically for
    use when training the adversarial generator.
    Parameters
    ----------
    batch_size : int
        Size of the batches generated by the generators returned by this method.
    preproc_size : list of ints
        Shape of each sample after pre-processing.

    Returns
    ----------
    Two generators: the first is for generating training data set batches, and the second is for generating test set
    batches.
    """
    if preproc_size is None:
        preproc_size = [28, 28, 1]
    (dataTrain, labelsTrain, dataTest, labelsTest) = load_HDF5()

    def preprocess_training_images(images, size):
        results = np.ndarray([images.shape[0]] + size, np.uint8)
        for idx in range(images.shape[0]):
            distorted = Preproc.randomFlipH(images[idx])
            distorted = Preproc.randomShift(distorted, rng=4)
            results[idx] = distorted.reshape([28, 28, 1])

        return results

    def preprocess_test_images(images, size):
        results = np.ndarray([images.shape[0]] + size, np.uint8)
        for idx in range(images.shape[0]):
            distorted = images[idx]
            distorted = Preproc.centerCrop(distorted, size)
            results[idx] = distorted

        return results

    def generate_batch(training):
        if training:
            generator = generate_data_label_pair(dataTrain, labelsTrain)
        else:
            generator = generate_data_label_pair(dataTest, labelsTest)

        while True:
            batchImages = []
            batchLabels = []
            batchTargets = []
            for _ in range(batch_size):
                images, labels = next(generator)
                batchImages.append(images)
                batchLabels.append(labels)
                batchTargets.append(random.randint(0, 9))

            if training:
                batchImages = preprocess_training_images(np.concatenate(batchImages, axis=0), preproc_size)
            else:
                batchImages = preprocess_test_images(np.concatenate(batchImages, axis=0), preproc_size)
            batchLabels = np.concatenate(batchLabels, axis=0)
            batchTargets = np.array(batchTargets)

            yield batchImages, batchLabels, batchTargets

    return generate_batch(training=True), generate_batch(training=False)


HParamMNIST = {'BatchSize': 200,
               'LearningRate': 1e-3,
               'MinLearningRate': 1e-5,
               'DecayAfter': 300,
               'ValidateAfter': 300,
               'TestSteps': 50,
               'TotalSteps': 40000}


class NetMNIST(Nets.Net):

    def __init__(self, image_shape, HParam=HParamMNIST):
        Nets.Net.__init__(self)

        self._init = False
        self._HParam = HParam
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)

        with self._graph.as_default():
            self._ifTest = tf.Variable(False, name='ifTest', trainable=False, dtype=tf.bool)
            self._step = tf.Variable(0, name='step', trainable=False, dtype=tf.int32)
            self._phaseTrain = tf.assign(self._ifTest, False)
            self._phaseTest = tf.assign(self._ifTest, True)

            # Inputs
            self._images = tf.placeholder(dtype=tf.float32, shape=[None] + image_shape,
                                          name='CIFAR10_images')
            self._labels = tf.placeholder(dtype=tf.int64, shape=[None],
                                          name='CIFAR10_labels_class')

            # Net
            self._body = self.body(self._images)
            self._inference = self.inference(self._body)
            self._accuracy = tf.reduce_mean(tf.cast(tf.equal(self._inference, self._labels), tf.float32))
            self._loss = self.lossClassify(self._body, self._labels)
            self._loss = 0
            self._updateOps = []
            for elem in self._layers:
                if len(elem.losses) > 0:
                    for tmp in elem.losses:
                        self._loss += tmp
            for elem in self._layers:
                if len(elem.updateOps) > 0:
                    for tmp in elem.updateOps:
                        self._updateOps.append(tmp)
            print(self.summary)
            print("\n Begin Training: \n")

            # Saver
            self._saver = tf.train.Saver(max_to_keep=5)

    def preproc(self, images):
        # Preprocessings
        casted = tf.cast(images, tf.float32)
        standardized = tf.identity(casted / 127.5 - 1.0, name='training_standardized')

        return standardized

    def body(self, images):
        # Preprocessings
        standardized = self.preproc(images)
        # Body
        net = Nets.VanillaNN(standardized, self._step, self._ifTest, self._layers)
        # net = Nets.LogisticRegression(standardized, self._step, self._ifTest, self._layers)
        # net = Nets.SimpleV1C(standardized, self._step, self._ifTest, self._layers)

        class10 = Layers.FullyConnected(net.output, outputSize=10, weightInit=Layers.XavierInit, wd=1e-4,
                                        biasInit=Layers.ConstInit(0.0),
                                        activation=Layers.Linear,
                                        name='FC_Coarse', dtype=tf.float32)
        self._layers.append(class10)

        return class10.output

    def inference(self, logits):
        return tf.argmax(logits, axis=-1, name='inference')

    def lossClassify(self, logits, labels, name='cross_entropy'):
        net = Layers.CrossEntropy(logits, labels, name=name)
        self._layers.append(net)
        return net.output

    def train(self, training_data_generator, test_data_generator, path_load=None, path_save=None):
        with self._graph.as_default():
            self._lr = tf.train.exponential_decay(self._HParam['LearningRate'],
                                                  global_step=self._step,
                                                  decay_steps=self._HParam['DecayAfter'] * 10,
                                                  decay_rate=0.30) + self._HParam['MinLearningRate']
            self._optimizer = tf.train.AdamOptimizer(self._lr, epsilon=1e-8).minimize(self._loss,
                                                                                      global_step=self._step)
            # Initialize all
            self._sess.run(tf.global_variables_initializer())

            if path_load is not None:
                self.load(path_load)

            self.evaluate(test_data_generator)
            #             self.sample(genTest)

            self._sess.run([self._phaseTrain])
            if path_save is not None:
                self.save(path_save)
            for _ in range(self._HParam['TotalSteps']):

                data, label = next(training_data_generator)

                loss, accu, step, _ = \
                    self._sess.run([self._loss,
                                    self._accuracy, self._step, self._optimizer],
                                   feed_dict={self._images: data,
                                              self._labels: label})
                self._sess.run(self._updateOps)
                print('\rStep: ', step,
                      '; L: %.3f' % loss,
                      '; A: %.3f' % accu,
                      end='')

                if step % self._HParam['ValidateAfter'] == 0:
                    self.evaluate(test_data_generator)
                    if path_save is not None:
                        self.save(path_save)
                    self._sess.run([self._phaseTrain])

    def evaluate(self, test_data_generator, path=None):
        if path is not None:
            self.load(path)

        totalLoss = 0.0
        totalAccu = 0.0
        self._sess.run([self._phaseTest])
        for _ in range(self._HParam['TestSteps']):
            data, label = next(test_data_generator)
            loss, accu = \
                self._sess.run([self._loss,
                                self._accuracy],
                               feed_dict={self._images: data,
                                          self._labels: label})
            totalLoss += loss
            totalAccu += accu
        totalLoss /= self._HParam['TestSteps']
        totalAccu /= self._HParam['TestSteps']
        print('\nTest: Loss: ', totalLoss,
              '; Accu: ', totalAccu)

    def infer(self, images):

        self._sess.run([self._phaseTest])

        return self._sess.run(self._inference, feed_dict={self._images: images})

    def save(self, path):
        self._saver.save(self._sess, path, global_step=self._step)

    def load(self, path):
        self._saver.restore(self._sess, path)


if __name__ == '__main__':
    net = NetMNIST([28, 28, 1])  # 8
    batchTrain, batchTest = get_data_generators(batch_size=HParamMNIST['BatchSize'])
    net.train(batchTrain, batchTest, path_save='./ClassifyMNIST/netmnist.ckpt')
# The best configuration is 64 features and 8 middle layers
