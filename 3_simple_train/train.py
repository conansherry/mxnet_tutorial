import sys, os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'

import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
import mxnet as mx

from create_network import get_symbol
from data import FileIter

if __name__ == "__main__":

    batch_size = 32
    lr = 0.001
    beta1 = 0.5

    dataiter = FileIter(r'E:\clean_data', batch_size)
    network = get_symbol()
    ctx = mx.gpu(1)

    # =============module network=============
    mod_network = mx.mod.Module(symbol=network, data_names=('data',), label_names=('l2_label',), context=ctx)
    mod_network.bind(data_shapes=dataiter.provide_data, label_shapes=dataiter.provide_label)
    mod_network.init_params(initializer=mx.init.Normal(0.02))
    mod_network.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
        })

    # create eval_metrix
    eval_metric = mx.metric.create('rmse')

    data_name = dataiter.data_name
    label_name = dataiter.label_name

    for epoch in range(10000):
        dataiter.reset()
        for t, batch in enumerate(dataiter):
            mod_network.forward(batch, is_train=True)
            mod_network.backward()
            mod_network.update()
            mod_network.update_metric(eval_metric, batch.label)

            print('epoch:', epoch, 'iter:', t, 'metric:', eval_metric.get())

        mod_network.save_params('model_%04d.params' % epoch)