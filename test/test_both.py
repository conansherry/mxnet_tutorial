import numpy as np
import caffe
import cv2
import mxnet as mx
import time

def get_data(image):
    image = image.transpose(2, 0, 1)
    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    return image

if __name__ == '__main__':
    ##################
    prototxt = r'D:\Workspace\image_segmentation\person_seg\person_seg_v6_deploy.prototxt'
    caffemodel = r'D:\Workspace\image_segmentation\person_seg\model\person_seg_v6_finetune_rotate360_iter_40000.caffemodel'

    caffe.set_mode_cpu()

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    ########################
    model_previx = 'model_pascal/resnet-29-360'
    epoch = 102
    ctx = mx.gpu(1)

    fcnxs, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_previx, epoch)

    fcnxs_args = {k: v.as_in_context(ctx) for k, v in fcnxs_args.items()}
    fcnxs_auxs = {k: v.as_in_context(ctx) for k, v in fcnxs_auxs.items()}
    mod = mx.mod.Module(symbol=fcnxs, context=ctx)
    #######################

    video_h = 480
    video_w = 640
    max_len = 160.0
    use_camera = True
    img_path = r'E:\image_seg_data\JPEGImages\COCO_124931.jpg'

    if not use_camera:
        test_im = cv2.imread(img_path)
        video_h = test_im.shape[0]
        video_w = test_im.shape[1]

    if video_h > video_w:
        ratio = max_len / video_h
        new_video_h = int(max_len)
        new_video_w = int(ratio * video_w)
    else:
        ratio = max_len / video_w
        new_video_w = int(max_len)
        new_video_h = int(ratio * video_h)

    del fcnxs_args['data']
    del fcnxs_args['softmax_label']
    del fcnxs_args['second_softmax_label']
    del fcnxs_args['third_softmax_label']
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, new_video_h, new_video_w))], force_rebind=True)
    mod.set_params(fcnxs_args, fcnxs_auxs, allow_missing=True)

    cap = cv2.VideoCapture(0)
    ori_bg = cv2.imread('bg.jpg')

    while True:
        ret, im = cap.read()
        # im = im.transpose((1, 0, 2))
        if not use_camera:
            im = test_im
        ori_im = np.copy(im)
        ori_im = cv2.resize(ori_im, (video_w, video_h))
        im = cv2.resize(im, (new_video_w, new_video_h))

        image = get_data(im)

        ################################
        start_time = time.time()

        net.blobs['data'].reshape(*(image.shape))
        net.reshape()
        forward_kwargs = {'data':image.astype(np.float32)}
        blobs_out = net.forward(**forward_kwargs)
        print("-[caffe]-- %s seconds ---" % (time.time() - start_time))

        output = blobs_out['softmax_score']

        ###########################
        start_time = time.time()
        mod.forward(mx.io.DataBatch([mx.nd.array(image)]))
        print("-[mxnet]-- %s seconds ---" % (time.time() - start_time))
        output = mod.get_outputs()[2]

        ###########################