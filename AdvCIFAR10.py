import random
import numpy as np
import tensorflow as tf
gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_logical_device_configuration(
    gpu,
    [tf.config.LogicalDeviceConfiguration(memory_limit=7200)])
    
import matplotlib.pyplot as plt

import Layers
import Nets
import CIFAR10

wd = 1e-4
NoiseRange = 10.0

def preproc(images):
    # Preprocessings
    casted        = tf.cast(images, tf.float32)
    standardized  = tf.identity(casted / 127.5 - 1.0)
        
    return standardized

def Generator(images, targets, numSubnets, step, ifTest, layers):   
    net = Layers.DepthwiseConv2D(preproc(images), convChannels=3*16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='G_DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='G_SepConv96', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layers.Conv2D(net.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='G_SepConv192Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='G_SepConv192a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        name='G_SepConv192b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='G_SepConv384Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='G_ReLU384')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='G_SepConv384a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='G_SepConv384b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='G_SepConv768Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='G_ReLU768')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='G_SepConv768a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='G_SepConv768b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='G_ReLU11024')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=1024, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='G_SepConv1024', dtype=tf.float32)
    layers.append(net)
    net = Layers.DeConv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        reuse=tf.compat.v1.AUTO_REUSE, name='G_DeConv192', dtype=tf.float32)
    layers.append(net)
    subnets = []
    for idx in range(numSubnets): 
        subnet = Layers.DeConv2D(net.output, convChannels=64, \
                            convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            reuse=tf.compat.v1.AUTO_REUSE, name='G_DeConv96_'+str(idx), dtype=tf.float32)
        layers.append(subnet)
        subnet = Layers.DeConv2D(subnet.output, convChannels=32, \
                            convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            reuse=tf.compat.v1.AUTO_REUSE, name='G_DeConv48_'+str(idx), dtype=tf.float32)
        layers.append(subnet)
        subnet = Layers.Conv2D(subnet.output, convChannels=3, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            reuse=tf.compat.v1.AUTO_REUSE, name='G_SepConv3_'+str(idx), dtype=tf.float32)
        layers.append(subnet)
        subnets.append(tf.expand_dims(subnet.output, axis=-1))
    subnets = tf.concat(subnets, axis=-1)
    weights = Layers.FullyConnected(tf.one_hot(targets, 10), outputSize=numSubnets, weightInit=Layers.XavierInit, wd=0.0, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.Softmax, \
                                reuse=tf.compat.v1.AUTO_REUSE, name='G_WeightsMoE', dtype=tf.float32)
    layers.append(weights)
    moe = tf.transpose(a=tf.transpose(a=subnets, perm=[1, 2, 3, 0, 4]) * weights.output, perm=[3, 0, 1, 2, 4])
    noises = (tf.nn.tanh(tf.reduce_sum(input_tensor=moe, axis=-1)) - 0.5) * NoiseRange * 2
    print('Shape of Noises: ', noises.shape)
    
    return noises

def PredictorSimpleNet(images, step, ifTest, layers):
    net = Layers.DepthwiseConv2D(preproc(tf.clip_by_value(images, 0, 255)), convChannels=3*16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv96', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layers.Conv2D(net.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv192Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv192a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        name='SepConv192b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv384Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU384')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv384a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv384b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv768Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU768')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv768a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv768b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU11024')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=1024, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='SepConv1024', dtype=tf.float32)
    layers.append(net)
#     net = Layers.SepConv2D(net.output, convChannels=1536, \
#                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
#                            convInit=Layers.XavierInit, convPadding='SAME', \
#                            biasInit=Layers.ConstInit(0.0), \
#                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
#                            activation=Layers.ReLU, \
#                            name='SepConv1536', dtype=tf.float32)
#     layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    logits = Layers.FullyConnected(net.output, outputSize=10, weightInit=Layers.XavierInit, wd=wd, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.Linear, \
                                reuse=tf.compat.v1.AUTO_REUSE, name='P_FC_classes', dtype=tf.float32)
    layers.append(logits)
    
    return logits.output

def PredictorSimpleNetG(images, step, ifTest, layers):
    net = Layers.DepthwiseConv2D(preproc(tf.clip_by_value(images, 0, 255)), convChannels=3*16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='DepthwiseConv3x16', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv96', dtype=tf.float32)
    
    toadd = Layers.Conv2D(net.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv192Shortcut', dtype=tf.float32)
    
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv192a', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        name='SepConv192b', dtype=tf.float32)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv384Shortcut', dtype=tf.float32)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU384')
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv384a', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv384b', dtype=tf.float32)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv768Shortcut', dtype=tf.float32)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU768')
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv768a', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv768b', dtype=tf.float32)
    
    added = toadd.output + net.output
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU11024')
    net = Layers.SepConv2D(net.output, convChannels=1024, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='SepConv1024', dtype=tf.float32)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    logits = Layers.FullyConnected(net.output, outputSize=10, weightInit=Layers.XavierInit, wd=wd, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.Linear, \
                                reuse=tf.compat.v1.AUTO_REUSE, name='P_FC_classes', dtype=tf.float32)
    
    return logits.output


def PredictorSmallNet(images, step, ifTest, layers):
    normalised_images = preproc(tf.clip_by_value(images, 0, 255))
    net = Nets.SmallNet(normalised_images, step, ifTest, layers)
    logits = Layers.FullyConnected(net.output, outputSize=10, weightInit=Layers.XavierInit, wd=wd,
                                   biasInit=Layers.ConstInit(0.0), activation=Layers.Linear, reuse=tf.compat.v1.AUTO_REUSE,
                                   name='P_FC_classes', dtype=tf.float32)
    layers.append(logits)

    return logits.output


def PredictorSmallNetG(images, step, ifTest):
    normalised_images = preproc(tf.clip_by_value(images, 0, 255))
    # pass empty list for layers, for the predictorG, we do not want to add the layers to the list
    net = Nets.SmallNet(normalised_images, step, ifTest, [])
    logits = Layers.FullyConnected(net.output, outputSize=10, weightInit=Layers.XavierInit, wd=wd,
                                   biasInit=Layers.ConstInit(0.0), activation=Layers.Linear,
                                   reuse=tf.compat.v1.AUTO_REUSE,
                                   name='P_FC_classes', dtype=tf.float32)

    return logits.output

def PredictorConcatNet(images, step, ifTest, layers):
    normalised_images = preproc(tf.clip_by_value(images, 0, 255))
    net = Nets.ConcatNet(normalised_images, step, ifTest, layers)
    logits = Layers.FullyConnected(net.output, outputSize=10, weightInit=Layers.XavierInit, wd=wd,
                                   biasInit=Layers.ConstInit(0.0), activation=Layers.Linear, reuse=tf.compat.v1.AUTO_REUSE,
                                   name='P_FC_classes', dtype=tf.float32)
    layers.append(logits)

    return logits.output


def PredictorConcatNetG(images, step, ifTest, num_middle):
    normalised_images = preproc(tf.clip_by_value(images, 0, 255))
    # pass empty list for layers, for the predictorG, we do not want to add the layers to the list
    net = Nets.ConcatNet(normalised_images, step, ifTest, [])
    logits = Layers.FullyConnected(net.output, outputSize=10, weightInit=Layers.XavierInit, wd=wd,
                                   biasInit=Layers.ConstInit(0.0), activation=Layers.Linear,
                                   reuse=tf.compat.v1.AUTO_REUSE,
                                   name='P_FC_classes', dtype=tf.float32)

    return logits.output
    
def PredictorXception(images, step, ifTest, layers, num_middle):
    normalised_images = preproc(tf.clip_by_value(images, 0, 255))
    net = Nets.Xcpetion(normalised_images, step, ifTest, layers, num_middle)
    logits = Layers.FullyConnected(net.output, outputSize=10, weightInit=Layers.XavierInit, wd=wd,
                                   biasInit=Layers.ConstInit(0.0), activation=Layers.Linear, reuse=tf.compat.v1.AUTO_REUSE,
                                   name='P_FC_classes', dtype=tf.float32)
    layers.append(logits)

    return logits.output


def PredictorXceptionG(images, step, ifTest, num_middle):
    normalised_images = preproc(tf.clip_by_value(images, 0, 255))
    # pass empty list for layers, for the predictorG, we do not want to add the layers to the list
    net = Nets.Xcpetion(normalised_images, step, ifTest, [], num_middle)
    logits = Layers.FullyConnected(net.output, outputSize=10, weightInit=Layers.XavierInit, wd=wd,
                                   biasInit=Layers.ConstInit(0.0), activation=Layers.Linear,
                                   reuse=tf.compat.v1.AUTO_REUSE,
                                   name='P_FC_classes', dtype=tf.float32)

    return logits.output

HParamCIFAR10 = {'BatchSize': 100,
                 'NumSubnets': 10, 
                 'NumPredictor': 1, 
                 'NumGenerator': 1, 
                 'NoiseDecay': 1e-5, 
                 'LearningRate': 1e-3, 
                 'MinLearningRate': 2 * 1e-5, 
                 'DecayAfter': 300,
                 'ValidateAfter': 300,
                 'TestSteps': 50,
                 'TotalSteps': 30000}

class NetCIFAR10(Nets.Net):
    
    def __init__(self, shapeImages, enemy, numMiddle=2, HParam=HParamCIFAR10):
        Nets.Net.__init__(self) 
        
        self._init = False
        self._numMiddle    = numMiddle
        self._HParam       = HParam
        self._graph        = tf.Graph()
        self._sess         = tf.compat.v1.Session(graph=self._graph)
        self._enemy        = enemy
        
        with self._graph.as_default(): 
            self._ifTest        = tf.Variable(False, name='ifTest', trainable=False, dtype=tf.bool)
            self._step          = tf.Variable(0, name='step', trainable=False, dtype=tf.int32)
            self._phaseTrain    = tf.compat.v1.assign(self._ifTest, False)
            self._phaseTest     = tf.compat.v1.assign(self._ifTest, True)
            
            # Inputs
            self._images = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self._HParam['BatchSize']]+shapeImages, \
                                          name='CIFAR10_images')
            self._labels = tf.compat.v1.placeholder(dtype=tf.int64, shape=[self._HParam['BatchSize']], \
                                          name='CIFAR10_labels')
            self._targets = tf.compat.v1.placeholder(dtype=tf.int64, shape=[self._HParam['BatchSize']], \
                                          name='CIFAR10_targets')
            
            # Net
            with tf.compat.v1.variable_scope('Generator', reuse=tf.compat.v1.AUTO_REUSE) as scope: 
                self._generator = Generator(self._images, self._targets, self._HParam['NumSubnets'], self._step, self._ifTest, self._layers)
            self._noises = self._generator
            self._adversary = self._noises + self._images
            with tf.compat.v1.variable_scope('Predictor', reuse=tf.compat.v1.AUTO_REUSE) as scope: 
                self._predictor = PredictorXception(self._images, self._step, self._ifTest, self._layers, numMiddle)
                self._predictorG = PredictorXceptionG(self._adversary, self._step, self._ifTest, numMiddle)
            self._inference = self.inference(self._predictor)
            self._accuracy = tf.reduce_mean(input_tensor=tf.cast(tf.equal(self._inference, self._labels), tf.float32))
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
            self._lossPredictor = self.lossClassify(self._predictor, self._labels, name='lossP') + self._loss
            self._lossGenerator = self.lossClassify(self._predictorG, self._targets, name='lossG') + self._HParam['NoiseDecay'] * tf.reduce_mean(input_tensor=tf.norm(tensor=self._noises)) + self._loss
            print(self.summary)
            print("\n Begin Training: \n")
                    
            # Saver
            self._saver = tf.compat.v1.train.Saver(max_to_keep=5)
        
    def preproc(self, images):
        # Preprocessings
        casted        = tf.cast(images, tf.float32)
        standardized  = tf.identity(casted / 127.5 - 1.0, name='training_standardized')
            
        return standardized
        
    def inference(self, logits):
        return tf.argmax(input=logits, axis=-1, name='inference')
    
    def lossClassify(self, logits, labels, name='cross_entropy'):
        net = Layers.CrossEntropy(logits, labels, name=name)
        self._layers.append(net)
        return net.output
    
    def train(self, genTrain, genTest, pathLoad=None, pathSave=None):
        with self._graph.as_default(): 
            self._lr = tf.compat.v1.train.exponential_decay(self._HParam['LearningRate'], \
                                                global_step=self._step, \
                                                decay_steps=self._HParam['DecayAfter'], \
                                                decay_rate=0.95) + self._HParam['MinLearningRate']
            #self._lr = tf.Variable(self._HParam['LearningRate'], trainable=False)
            #self._lrDecay1 = tf.compat.v1.assign(self._lr, self._lr * 0.1)
            self._stepInc = tf.compat.v1.assign(self._step, self._step+1)
            self._varsG = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
            self._varsP = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Predictor')
            self._optimizerG = tf.compat.v1.train.AdamOptimizer(self._lr, epsilon=1e-8)
            self._optimizerP = tf.compat.v1.train.AdamOptimizer(self._lr, epsilon=1e-8).minimize(self._lossPredictor, var_list=self._varsP)
            gradientsG = self._optimizerG.compute_gradients(self._lossGenerator, var_list=self._varsG)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradientsG]
            self._optimizerG = self._optimizerG.apply_gradients(capped_gvs)
#            self._optimizerG = tf.train.AdamOptimizer(self._lr, epsilon=1e-8).minimize(self._lossGenerator, var_list=self._varsG)
#            self._optimizerP = tf.train.AdamOptimizer(self._lr, epsilon=1e-8).minimize(self._lossPredictor, var_list=self._varsP)
            
            # Initialize all
            self._sess.run(tf.compat.v1.global_variables_initializer())

            if pathLoad is not None:
                #tf.compat.v1.disable_eager_execution()
                self.load(pathLoad)
                #tf.compat.v1.enable_eager_execution()
            else:
                print('Warming up. ')
                for idx in range(300):
                    data, label, target = next(genTrain)
                    label = np.array(self._enemy.infer(data))
                    loss, accu, _ = \
                        self._sess.run([self._lossPredictor, \
                                        self._accuracy, self._optimizerP], \
                                        feed_dict={self._images: data, \
                                                    self._labels: label, \
                                                self._targets: target})
                    print('\rPredictor => Step: ', idx-300, \
                            '; Loss: %.3f'% loss, \
                            '; Accuracy: %.3f'% accu, \
                            end='')

                warmupAccu = 0.0
                for idx in range(50):
                    data, label, target = next(genTest)
                    label = np.array(self._enemy.infer(data))
                    loss, accu, _ = \
                        self._sess.run([self._lossPredictor, \
                                        self._accuracy, self._optimizerP], \
                                        feed_dict={self._images: data, \
                                                    self._labels: label, \
                                                self._targets: target})
                    warmupAccu += accu / 50
                print('\nWarmup Accuracy: ', warmupAccu)
#

            self.evaluate(genTest)
#             self.sample(genTest)
            
            self._sess.run([self._phaseTrain])
            if pathSave is not None:
                self.save(pathSave)
            
            globalStep = 0
            
            while globalStep < self._HParam['TotalSteps']: 
                
                self._sess.run(self._stepInc)
                
                for _ in range(self._HParam['NumPredictor']): 
                    data, label, target = next(genTrain)
                    adversary = self._sess.run(self._adversary, \
                                               feed_dict={self._images: data, \
                                                          self._labels: label, \
                                                          self._targets: target})
                    data = data + (np.random.rand(self._HParam['BatchSize'], 32, 32, 3) - 0.5) * 2 * NoiseRange
                    label = self._enemy.infer(data)
                    loss, accu, globalStep, _ = \
                        self._sess.run([self._lossPredictor, \
                                        self._accuracy, self._step, self._optimizerP], \
                                        feed_dict={self._images: data, \
                                                   self._labels: label, \
                                                   self._targets: target})
                    print('\rSimulator => Step: ', globalStep, \
                                '; Loss: %.3f'% loss, \
                                '; Accuracy: %.3f'% accu, \
                                end='')
                    label = self._enemy.infer(adversary)
                    loss, accu, globalStep, _ = \
                        self._sess.run([self._lossPredictor, \
                                        self._accuracy, self._step, self._optimizerP], \
                                        feed_dict={self._images: adversary, \
                                                   self._labels: label, \
                                                   self._targets: target})

                    self.simulator_loss_history.append(loss)
                    self.simulator_accuracy_history.append(accu)
                    print('\rSimulator => Step: ', globalStep, \
                                '; Loss: %.3f'% loss, \
                                '; Accuracy: %.3f'% accu, \
                                end='')
                    
                for _ in range(self._HParam['NumGenerator']): 
                    data, label, target = next(genTrain)
                    refs = self._enemy.infer(data)
                    for idx in range(data.shape[0]):
                        if refs[idx] == target[idx]: 
                            tmp = random.randint(0, 9)
                            while tmp == refs[idx]: 
                                tmp = random.randint(0, 9)
                            target[idx] = tmp
                    loss, adversary, globalStep, _ = \
                        self._sess.run([self._lossGenerator, \
                                        self._adversary, self._step, self._optimizerG], \
                                        feed_dict={self._images: data, \
                                                self._labels: refs, \
                                                self._targets: target})
                    results = self._enemy.infer(adversary)
                    accu = np.mean(target==results)
                    fullrate = np.mean(refs!=results)

                    self.generator_loss_history.append(loss)
                    self.generator_accuracy_history.append(accu)
                    print('\rGenerator => Step: ', globalStep, \
                            '; Loss: %.3f'% loss, \
                            '; TFR: %.3f'% accu, \
                            '; UFR: %.3f'% fullrate, \
                            end='')
                
                if globalStep % self._HParam['ValidateAfter'] == 0: 
                    self.evaluate(genTest)
                    data, label, target = next(genTest)
                    adversary = \
                        self._sess.run(self._adversary, \
                                        feed_dict={self._images: data, \
                                                self._labels: label, \
                                                self._targets: target})
                    refs = self._enemy.infer(data)
                    results = self._enemy.infer(adversary)
                    # print(np.max(adversary-data))
                    # print(np.min(adversary-data))
                    # print((adversary-data)[1])
                    # print(list(zip(label, refs, results, target)))
                    if pathSave is not None:
                        self.save(pathSave)
                        np.savez("./AttackCIFAR10/training_history", self.simulator_loss_history,
                                 self.simulator_accuracy_history, self.generator_loss_history,
                                 self.generator_accuracy_history, self.test_loss_history, self.test_accuracy_history)

                    self._sess.run([self._phaseTrain])
                
                #if globalStep == 7200 or globalStep == 10200: 
                #    self._sess.run(self._lrDecay1)
                #    print('Learning rate decayed. ')
                
    def evaluate(self, genTest, path=None):
        if path is not None:
            self.load(path)
        
        totalLoss  = 0.0
        totalAccu  = 0.0
        totalFullRate  = 0.0
        self._sess.run([self._phaseTest])  
        for _ in range(self._HParam['TestSteps']): 
            data, label, target = next(genTest)
            refs = self._enemy.infer(data)
            for idx in range(data.shape[0]):
                if refs[idx] == target[idx]: 
                    tmp = random.randint(0, 9)
                    while tmp == refs[idx]: 
                        tmp = random.randint(0, 9)
                    target[idx] = tmp
            loss, adversary = \
                self._sess.run([self._lossGenerator, \
                                self._adversary], \
                                feed_dict={self._images: data, \
                                           self._labels: refs, \
                                           self._targets: target})
            adversary = adversary.clip(0, 255).astype(np.uint8)
            results = self._enemy.infer(adversary)
            accu = np.mean(target==results)
            fullrate = np.mean(refs!=results)
            totalLoss += loss
            totalAccu += accu
            totalFullRate += fullrate
        totalLoss /= self._HParam['TestSteps']
        totalAccu /= self._HParam['TestSteps']
        totalFullRate /= self._HParam['TestSteps']

        self.test_loss_history.append(totalLoss)
        self.test_accuracy_history.append(totalAccu)
        print('\nTest: Loss: ', totalLoss, \
              '; TFR: ', totalAccu,
              '; UFR: ', totalFullRate)

    def sample(self, genTest, path=None):
        if path is not None:
            self.load(path)
            
        self._sess.run([self._phaseTest])  
        data, label, target = next(genTest)
        data, label, target = next(genTest)
        refs = self._enemy.infer(data)
        for idx in range(data.shape[0]):
            if refs[idx] == target[idx]: 
                tmp = random.randint(0, 9)
                while tmp == refs[idx]: 
                    tmp = random.randint(0, 9)
                target[idx] = tmp
        loss, adversary = \
            self._sess.run([self._lossGenerator, \
                            self._adversary], \
                            feed_dict={self._images: data, \
                                        self._labels: refs, \
                                        self._targets: target})
        adversary = adversary.clip(0, 255).astype(np.uint8)
        results = self._enemy.infer(adversary)
        
        
        for idx in range(10): 
            for jdx in range(3): 
                plt.subplot(10, 6, idx*6+jdx*2+1)
                plt.imshow(data[idx*3+jdx])
                plt.subplot(10, 6, idx*6+jdx*2+2)
                plt.imshow(adversary[idx*3+jdx])
                print([refs[idx*3+jdx], results[idx*3+jdx], target[idx*3+jdx]])
        plt.show()
       
    def plot(self, genTest, path=None): 
        if path is not None:
            self.load(path)
        
        data, label, target = next(genTest)
        
        tmpdata = []
        tmptarget = []
        
        for idx in range(10):
            while True: 
                jdx = 0
                while jdx < data.shape[0]:
                    if label[jdx] == idx: 
                        break
                    jdx += 1
                if jdx < data.shape[0]: 
                    break
                else: 
                    data, label, target = next(genTest)
            for ldx in range(10): 
                if ldx != idx: 
                    tmpdata.append(data[jdx][np.newaxis, :, :, :])
                    tmptarget.append(ldx)
        tmpdata = np.concatenate(tmpdata, axis=0)
        tmptarget = np.array(tmptarget)
            
        adversary = \
        self._sess.run(self._adversary, \
                        feed_dict={self._images: tmpdata, \
                                    self._targets: tmptarget})
        adversary = adversary.clip(0, 255).astype(np.uint8)
        
        kdx = 0
        for idx in range(10):
            jdx = 0
            while jdx < 10: 
                if jdx == idx: 
                    jdx += 1
                    continue
                plt.subplot(10, 10, idx*10+jdx+1)
                plt.imshow(adversary[kdx, :, :, 0], cmap='gray')
                plt.axis('off')
                jdx += 1
                kdx += 1
                
        plt.show()
    
    def save(self, path):
        self._saver.save(self._sess, path, global_step=self._step)
    
    def load(self, path):
        self._saver.restore(self._sess, path)
        self.load_training_history("./AttackCIFAR10/training_history")

if __name__ == '__main__':
    #tf.compat.v1.experimental.output_all_intermediates(True)
    enemy = CIFAR10.NetCIFAR10([32, 32, 3], 2)
    tf.compat.v1.disable_eager_execution()
    enemy.load('./ClassifyCIFAR10/netcifar10.ckpt-29701')
    tf.compat.v1.enable_eager_execution()

    net = NetCIFAR10([32, 32, 3], enemy=enemy, numMiddle=2)
    batchTrain, batchTest = CIFAR10.generatorsAdv(BatchSize=HParamCIFAR10['BatchSize'], preprocSize=[32, 32, 3])
    
    #while True: 
    #    net.plot(batchTest, './AttackCIFAR10/netcifar10.ckpt-18600')
        
        
    net.train(batchTrain, batchTest, pathSave='./AttackCIFAR10/netcifar10.ckpt')
    net.plot_training_history("Adversarial CIFAR10")
    #net.evaluate(batchTest, './AttackCIFAR10/netcifar10.ckpt-16500')
    #net.sample(batchTest, './AttackCIFAR10/netcifar10.ckpt-6900')

    # Loss:  0.8548992574214935 ; TFR:  0.74890625 ; UFR:  0.86390625 after 30000 steps with SmallNet as target and
    # SimpleNet as simulator, lower than TFR of 0.843 in paper. Plateaued after 12000 steps.
    # Loss:  0.542461564540863 ; TFR:  0.8415625 ; UFR:  0.91078125 after 27000 steps with SmallNet as target and
    # SimpleNet as simulator, almost the same as in the paper, after using exponential learning rate decay! Hit 0.82
    # TFR after 19200 steps, and then increased very slowly.
    
    # SmallNet as both simulator and target:
    # Loss:  0.6553342294692993; TFR: 0.80203125; UFR: 0.89640625 after 30000
    # Loss:  0.5618793880939483 ; TFR:  0.8396875 ; UFR:  0.91234375, with decay rate 0.95 and min learning step of 2 * 10^-5, close to 0.85 result in paper.
    # Test: Loss:  0.5125626558065415 ; TFR:  0.8568000000000001 ; UFR:  0.9262999999999999 after 30000 steps with decay rate 0.95, min learning rate 2 * 10^-5 and batch 200
    
    # ConcatNet simulator and SmallNet target
    # Test: Loss:  0.6571377271413803 ; TFR:  0.7969999999999997 ; UFR:  0.9055999999999998 after 29100 steps with batch 100 (batch 110 exhausts memory)
    
    # Cross Model Attack
    # SimpleV7->SimpleV7; Accu:  0.8017 ; FullRate:  0.8772000000000001
    # SimpleV7->Xception; Accu:  0.5671999999999999 ; FullRate:  0.7422000000000001
    # SimpleV7->SimpleV1C; Accu:  0.5924999999999999 ; FullRate:  0.7773
    # SimpleV7->SimpleV3; Accu:  0.29949999999999993 ; FullRate:  0.54
    
    # SimpleV1C->SimpleV1C; Accu:  0.8438000000000001 ; FullRate:  0.9072000000000002
    # SimpleV1C->SimpleV7; Accu:  0.5448000000000001 ; FullRate:  0.7070999999999998
    # SimpleV1C->SimpleV3; Accu:  0.2624 ; FullRate:  0.5051000000000002
    # SimpleV1C->Xception; Accu:  0.4848999999999999 ; FullRate:  0.6812
    
    # SimpleV3->SimpleV1C; Accu:  0.5075000000000001 ; FullRate:  0.7004
    # SimpleV3->SimpleV7; Accu:  0.48120000000000007 ; FullRate:  0.664
    # SimpleV3->SimpleV3; Accu:  0.6988999999999999 ; FullRate:  0.8077000000000002
    # SimpleV3->Xception; Accu:  0.5119000000000001 ; FullRate:  0.6979999999999997
    
    # Xception->Xception: Accu:  0.8236999999999999 ; FullRate:  0.8931999999999997
    # Xception->SimpleV1C; Accu:  0.6172 ; FullRate:  0.7890999999999998
    # Xception->SimpleV7; Accu:  0.6298 ; FullRate:  0.7705000000000001
    # Xception->SimpleV3; Accu:  0.29150000000000004 ; FullRate:  0.5341999999999999
    
