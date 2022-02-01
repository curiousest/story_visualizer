from __future__ import division

import datetime
import getopt
import os
import pprint
import sys
from functools import partial

import caffe
import numpy as np
import PIL.Image
import scipy.ndimage as nd
from cStringIO import StringIO
from google.protobuf import text_format
from IPython.display import Image, clear_output, display

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist


def showarray(a, fmt="jpeg"):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


model_path = "/deepdream/caffe/models/bvlc_googlenet/"  # substitute your path here
net_fn = model_path + "deploy.prototxt"
param_fn = model_path + "bvlc_googlenet.caffemodel"

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open("tmp.prototxt", "w").write(str(model))

net = caffe.Classifier(
    "tmp.prototxt",
    param_fn,
    mean=np.float32([104.0, 116.0, 122.0]),  # ImageNet mean, training set dependent
    channel_swap=(2, 1, 0),
)  # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean["data"]


def deprocess(net, img):
    return np.dstack((img + net.transformer.mean["data"])[::-1])


def objective_L2(dst):
    dst.diff[:] = dst.data


def make_step(
    net,
    step_size=1.5,
    end="inception_4c/output",
    jitter=32,
    clip=True,
    objective=objective_L2,
):
    """Basic gradient ascent step."""

    src = net.blobs["data"]  # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter + 1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)  # apply jitter shift

    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size / np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)  # unshift image

    if clip:
        bias = net.transformer.mean["data"]
        src.data[:] = np.clip(src.data, -bias, 255 - bias)


def deepdream(
    net,
    base_img,
    save_image_function,
    print_every_step=False,
    iter_n=10,
    octave_n=4,
    octave_scale=1.4,
    end="inception_4c/output",
    clip=True,
    **step_params
):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n - 1):
        octaves.append(
            nd.zoom(octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale), order=1)
        )

    src = net.blobs["data"]
    detail = np.zeros_like(octaves[-1])  # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        src.reshape(1, 3, h, w)  # resize the network's input image size
        src.data[0] = octave_base + detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)

            # visualization
            vis = deprocess(net, src.data[0])
            if not clip:  # adjust image contrast if clipping is disabled
                vis = vis * (255.0 / np.percentile(vis, 99.98))
            showarray(vis)
            print(octave, i, end, vis.shape)
            if print_every_step or i == iter_n - 1:
                save_image_function(vis, filename="{}_{}.jpg".format(octave, i))
            clear_output(wait=True)

        # extract details produced on the current octave
        detail = src.data[0] - octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])


IMAGES_BASE_DIR = "/deepdream/deepdream/files/images/"


def get_image_objective(guide_image_filename, end="inception_3b/output"):
    filename = "{}themes/{}".format(IMAGES_BASE_DIR, guide_image_filename)
    guide = PIL.Image.open(filename)

    # shrink the size to 256 on the biggest edge
    if guide.size[0] > guide.size[1]:
        new_width = min(guide.size[0], 256)
        new_height = round(guide.size[1] * (new_width / guide.size[0]))
    else:
        new_height = min(guide.size[1], 256)
        new_width = round(guide.size[0] * (new_height / guide.size[1]))
    guide = guide.resize((int(new_width), int(new_height)))

    if filename[-4:] == ".png":
        guide = guide.convert("RGB")
    guide = np.float32(guide)
    h, w = guide.shape[:2]
    src, dst = net.blobs["data"], net.blobs[end]
    src.reshape(1, 3, h, w)
    src.data[0] = preprocess(net, guide)
    net.forward(end=end)
    guide_features = dst.data[0].copy()

    def custom_objective_guide(dst):
        x = dst.data[0].copy()
        y = guide_features
        ch = x.shape[0]
        x = x.reshape(ch, -1)
        y = y.reshape(ch, -1)
        A = x.T.dot(y)  # compute the matrix of dot-products with guide features
        dst.diff[0].reshape(ch, -1)[:] = y[
            :, A.argmax(1)
        ]  # select ones that match best

    return custom_objective_guide


def save_deepdream_image(image, filename, directory=""):
    new_image = PIL.Image.fromarray(image.astype("uint8"), "RGB")
    new_image.save("{}{}{}".format(IMAGES_BASE_DIR, directory, filename))


def generate_and_save_dream(
    image,
    folder_name,
    guide_image_filename=None,
    dream_params=None,
    extra_description="",
):
    base_dream_params = {
        "net": net,
        "iter_n": 4,
        "octave_n": 4,
        "octave_scale": 1.4,
        "end": "inception_4c/output",
        "clip": True,
        "step_size": 1.5,
        "jitter": 32,
        "objective": objective_L2,
    }
    if dream_params is not None:
        base_dream_params.update(dream_params)
    if guide_image_filename is not None:
        base_dream_params["objective"] = get_image_objective(guide_image_filename)
        base_dream_params["end"] = (
            dream_params.get("end") if dream_params else base_dream_params["end"]
        )
    directory = "{}/".format(folder_name)
    save_image_function = partial(save_deepdream_image, directory=directory)
    try:
        os.mkdir(IMAGES_BASE_DIR + directory)
    except OSError:
        pass
    start_dt = datetime.datetime.now()
    deepdream(
        base_img=image, save_image_function=save_image_function, **base_dream_params
    )
    end_dt = datetime.datetime.now()
    base_dream_params.update(
        {
            "filename": folder_name,
            "extra_description": extra_description,
            "running_time": str(end_dt - start_dt),
            "start_time": str(start_dt),
            "guide_image": guide_image_filename,
        }
    )
    with open("{}{}metadata.txt".format(IMAGES_BASE_DIR, directory), "w") as f:
        f.write(pprint.pformat(base_dream_params))


def main(argv):
    try:
        opts, args = getopt.getopt(
            argv, "hi:o:g:n:", ["ifile=", "ofile=", "gfile=", "n="]
        )
    except getopt.GetoptError as e:
        print(str(e))
        print(
            "deepdream.py -i <inputfile> -o <outputfolder> -g <guidefile> -n <iter_n>"
        )
        sys.exit(2)
    guidefile = None
    iter_n = None
    for opt, arg in opts:
        if opt == "-h":
            print(
                "deepdream.py -i <inputfile> -o <outputfolder> -g <guidefile> -n <iter_n>"
            )
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfolder = arg
        elif opt in ("-g", "--gfile"):
            guidefile = arg
        elif opt in ("-n", "--iter_n"):
            iter_n = int(arg)

    image_path = "/deepdream/deepdream/files/{}".format(inputfile)
    img = PIL.Image.open(image_path)
    if image_path[-4:] == ".png":
        img = img.convert("RGB")
    img = np.float32(img)
    dream_params = {}
    if guidefile is not None:
        dream_params["end"] = "inception_3b/output"
    dream_params["iter_n"] = iter_n if iter_n is not None else 4
    generate_and_save_dream(
        img,
        folder_name=outputfolder,
        guide_image_filename=guidefile,
        dream_params=dream_params,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
