import mxnet as mx

def residual_unit(data, num_filter, stride, dim_match, name, bn_mom=0.9, workspace=256, memonger=False):
    conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1), no_bias=True, workspace=workspace, name=name + '_conv1')
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True, workspace=workspace, name=name + '_conv2')
    bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True, workspace=workspace, name=name+'_sc')
        shortcut = mx.sym.BatchNorm(data=shortcut, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_shortcut_bn')
        shortcut = mx.sym.Activation(data=shortcut, act_type='relu', name=name + '_shortcut_relu')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return act2 + shortcut

def get_symbol(output_dim=192, bn_mom=0.9, workspace=256, memonger=False):

    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')

    body = mx.sym.Convolution(data=data, num_filter=24, kernel=(5, 5), stride=(2, 2), pad=(2, 2), no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, momentum=bn_mom, eps=2e-5, name='body_bn')
    body = mx.sym.Activation(data=body, act_type='relu', name='body_relu')

    body = residual_unit(body, 48, (2, 2), False, name='stage%d_unit%d' % (1, 1), workspace=workspace, memonger=memonger)

    body = residual_unit(body, 64, (2, 2), False,  name='stage%d_unit%d' % (2, 1), workspace=workspace,  memonger=memonger)

    body = residual_unit(body, 80, (2, 2), False, name='stage%d_unit%d' % (3, 1), workspace=workspace, memonger=memonger)

    body = residual_unit(body, 128,  (2, 2), False, name='stage%d_unit%d' % (4, 1), workspace=workspace, memonger=memonger)

    body = mx.sym.flatten(data=body)
    body = mx.sym.FullyConnected(data=body, num_hidden=512)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, momentum=bn_mom, eps=2e-5, name='fcbn')
    body = mx.sym.Activation(data=body, act_type='relu', name='fcrelu')

    body = mx.sym.FullyConnected(data=body, num_hidden=output_dim)

    body = mx.sym.LinearRegressionOutput(data=body, name='l2')

    return body

if __name__ == "__main__":
    network = get_symbol()

    output_name = network.get_internals().list_outputs()
    arg_shape, output_shape, aux_shape = network.get_internals().infer_shape(data=(1, 3, 128, 128))
    print zip(output_name, output_shape)

    tmp = mx.viz.plot_network(network, shape={'data': (1, 3, 128, 128)})
    tmp.view()