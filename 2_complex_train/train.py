import sys, os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'

import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
import mxnet as mx
from mxnet import optimizer as opt
from mxnet.optimizer import get_updater
from mxnet import metric

from create_network import get_symbol
from init import init_params
from data import FileIter

from collections import namedtuple
BatchEndParam = namedtuple('BatchEndParams', ['epoch', 'nbatch', 'eval_metric'])

if __name__ == "__main__":
    ctx = mx.gpu(1)
    network = get_symbol()
    model_prefix = 'facealignment'
    save_model_prefix = "facealignment"
    # _, network_args, network_auxs = mx.model.load_checkpoint(model_prefix, 1)
    network_args = {}
    network_auxs = {}
    network_args, network_auxs = init_params(ctx, network, network_args, network_auxs, (1, 3, 128, 128))

    dataiter = FileIter(r'E:\mxnet_tutorial_data')

    # prepare grad_params
    arg_shapes, out_shapes, aux_shapes = network.infer_shape(data=dataiter.provide_data[0][1])
    arg_names = network.list_arguments()
    grad_params = {}
    for name, shape in zip(arg_names, arg_shapes):
        if not (name.endswith('data') or name.endswith('label')):
            grad_params[name] = mx.nd.zeros(shape, ctx)

    # prepare aux_params
    aux_names = network.list_auxiliary_states()
    aux_params = {k: mx.nd.zeros(s, ctx) for k, s in zip(aux_names, aux_shapes)}

    # prepare optimizer
    optimizer = opt.create('adam', rescale_grad=(1.0/dataiter.get_batch_size()), **({'learning_rate': 0.01}))
    updater = get_updater(optimizer)

    # create eval_metrix
    eval_metric = metric.create('rmse')

    data_name = dataiter.data_name
    label_name = dataiter.label_name
    arg_params = network_args
    aux_params = network_auxs

    batch_callback = mx.callback.Speedometer(1, 10)
    epoch_callback = mx.callback.do_checkpoint(save_model_prefix)

    # begin training
    for epoch in range(10000):
        nbatch = 0
        dataiter.reset()
        eval_metric.reset()
        for data in dataiter:
            nbatch += 1
            label_shape = data[label_name].shape
            arg_params[data_name] = mx.nd.array(data[data_name], ctx)
            arg_params[label_name] = mx.nd.array(data[label_name], ctx)
            exector = network.bind(ctx, arg_params, args_grad=grad_params, grad_req='write', aux_states=aux_params)
            assert len(network.list_arguments()) == len(exector.grad_arrays)
            update_dict = {name: nd for name, nd in zip(network.list_arguments(), exector.grad_arrays) if nd is not None}
            output_dict = {}
            output_buff = {}
            for key, arr in zip(network.list_outputs(), exector.outputs):
                output_dict[key] = arr
                output_buff[key] = mx.nd.empty(arr.shape, ctx=mx.cpu())
            exector.forward(is_train=True)
            for key in output_dict:
                output_dict[key].copyto(output_buff[key])
            exector.backward()
            for key, arr in update_dict.items():
                updater(key, arr, arg_params[key])
            pred_shape = exector.outputs[0].shape
            label = mx.nd.array(data[label_name])
            pred = mx.nd.array(output_buff["l2_output"].asnumpy())
            eval_metric.update([label], [pred])
            exector.outputs[0].wait_to_read()
            batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch, eval_metric=eval_metric)
            batch_callback(batch_end_params)
        if epoch_callback is not None:
            epoch_callback(epoch, network, arg_params, aux_params)
        name, value = eval_metric.get()
        logging.info("                     --->Epoch[%d] Train-%s=%f", epoch, name, value)

        # evaluation
        # eval_data = dataiter
        # if eval_data:
        #     logger.info(" in eval process...")
        #     nbatch = 0
        #     eval_data.reset()
        #     eval_metric.reset()
        #     for data in eval_data:
        #         nbatch += 1
        #         label_shape = data[label_name].shape
        #         .arg_params[data_name] = mx.nd.array(data[data_name], ctx)
        #         arg_params[label_name] = mx.nd.array(
        #             data[label_name].reshape(label_shape[0], label_shape[1] * label_shape[2] * label_shape[3]),
        #             ctx)
        #         exector = network.bind(ctx, arg_params,
        #                                    args_grad=grad_params,
        #                                    grad_req='write',
        #                                    aux_states=aux_params)
        #         cpu_output_array = mx.nd.zeros(exector.outputs[0].shape)
        #         exector.forward(is_train=False)
        #         exector.outputs[0].copyto(cpu_output_array)
        #         pred_shape = cpu_output_array.shape
        #         label = mx.nd.array(data[label_name].reshape(label_shape[0], label_shape[1] * label_shape[2] * label_shape[3]))
        #         pred = mx.nd.array( cpu_output_array.asnumpy().reshape(pred_shape[0], pred_shape[1], pred_shape[2] * pred_shape[3]))
        #         eval_metric.update([label], [pred])
        #         exector.outputs[0].wait_to_read()
        #     name, value = eval_metric.get()
        #     logger.info('batch[%d] Validation-%s=%f', nbatch, name, value)