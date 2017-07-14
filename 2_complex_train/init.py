# pylint: skip-file
import mxnet as mx

def init_params(ctx, network_symbol, network_args_from, network_auxs_from, input_shape):
    new_args_names = network_symbol.list_arguments()
    new_auxs_names = network_symbol.list_auxiliary_states()
    new_args_shapes, _, new_auxs_shapes = network_symbol.infer_shape(data=input_shape)
    new_args_dict = dict(zip(new_args_names, new_args_shapes))
    new_auxs_dict = dict(zip(new_auxs_names, new_auxs_shapes))

    del new_args_dict['data']
    del new_args_dict['l2_label']

    res_args = {}
    res_auxs = {}

    new_auxs_keys = new_auxs_dict.keys()
    old_auxs_keys = network_auxs_from.keys()
    for k, v in network_auxs_from.items():
        if (v.context != ctx):
            res_auxs[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(res_auxs[k])
    has_init_value_args = list(set(new_auxs_keys).intersection(set(old_auxs_keys)))
    for key in has_init_value_args:
        if new_auxs_dict[key] == network_auxs_from[key].shape:
            res_auxs[key] = network_auxs_from[key].as_in_context(ctx)
        else:
            res_auxs[k] = mx.nd.zeros(new_auxs_dict[key], ctx)
    rest_auxs = list(set(new_auxs_keys).difference(set(old_auxs_keys)))
    for key in rest_auxs:
        res_auxs[key] = mx.nd.zeros(new_auxs_dict[key], ctx)

    init_tool = mx.initializer.Xavier()
    new_args_keys = new_args_dict.keys()
    old_args_keys = network_args_from.keys()
    has_init_value_args = list(set(new_args_keys).intersection(set(old_args_keys)))
    for key in has_init_value_args:
        if new_args_dict[key] == network_args_from[key].shape:
            res_args[key] = network_args_from[key].as_in_context(ctx)
        else:
            arr = mx.nd.zeros(new_args_dict[key], ctx)
            init_tool(mx.init.InitDesc(key), arr)
            res_args[key] = arr
    rest_args = list(set(new_args_keys).difference(set(old_args_keys)))
    for key in rest_args:
        arr = mx.nd.zeros(new_args_dict[key], ctx)
        init_tool(mx.init.InitDesc(key), arr)
        res_args[key] = arr

    return res_args, res_auxs