"""
Demo of CMR.

Note that CMR assumes that the object has been detected, so please use a picture of a bird that is centered and well cropped.

Sample usage:

python -m cmr.demo --name bird_net --num_train_epoch 500 --img_path cmr/demo_data/img1.jpg
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
from absl import flags, app
import numpy as np
import skimage.io as io
import pickle

import torch

from .nnutils import test_utils
from .nnutils import predictor as pred_util
from .utils import image as img_util
from .utils import mesh
from .utils import geometry as geom_utils


flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
# flags.DEFINE_integer('img_size', 224, 'image size the network was trained on.')
flags.DEFINE_integer('use_train', -1, 'Use data from CUB train data')


opts = flags.FLAGS


def preprocess_image(img_path, img_size=256):
    img = io.imread(img_path)[:,:,:-1] / 255.

    # Scale the max image size to be img_size
#     scale_factor = float(img_size) / np.max(img.shape[:2])
#     img, _ = img_util.resize_img(img, scale_factor)

    # Crop img_size x img_size from the center
#     center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # img center in (x, y)
#     center = center[::-1]
#     bbox = np.hstack([center - img_size / 2., center + img_size / 2.])

#     img = img_util.crop(img, bbox, bgval=1.)

    # Transpose the image to 3xHxW
    img = np.transpose(img, (2, 0, 1))
    print(img.shape)

    return img


def visualize(img, outputs, renderer):
    vert = outputs['verts'][0]
    print("vert:", vert.shape, vert[:2])
    faces_car = outputs['faces']
    print("faces_car:", faces_car.shape, faces_car[:2])
    
    
#     sp_verts, sp_faces = mesh.create_sphere(4)
#     print("created sphere")
#     verts_dense = geom_utils.project_verts_on_mesh(sp_verts, vert, faces_car) 
#     print("dense vert:", verts_dense.shape)
#     print("dense faces:", sp_faces.shape)
    
    
    vert_np = (torch.autograd.Variable(vert).data).cpu().numpy()
#     faces_np = (torch.autograd.Variable(faces_car).data).cpu().numpy()
    print("vertnp: ", vert_np[:3])
    with open('mesh_verts.pkl', 'wb') as handle:
        pickle.dump([vert_np, faces_car], handle)
    print("pickled out mesh")
        
    print("vert shape", outputs['verts'].shape)
    cam = outputs['cam_pred'][0]
    print("cam", cam)
    texture = outputs['texture'][0]
#     cam[-1] = 0
#     cam[-2] = 0
#     cam[-3] = 0
#     cam[-4] = 0
    
#     shape_pred = renderer(vert, cam)
    shape_pred = renderer.diff_vp(
        vert, cam, angle=0, axis=[0, 1, 0], texture=None, extra_elev=True)
    img_pred = renderer(vert, cam, texture=texture)

    # Different viewpoints.
    vp1 = renderer.diff_vp(
        vert, cam, angle=30, axis=[0, 1, 0], texture=texture, extra_elev=True)
    vp2 = renderer.diff_vp(
        vert, cam, angle=60, axis=[0, 1, 0], texture=texture, extra_elev=True)
    vp3 = renderer.diff_vp(
        vert, cam, angle=60, axis=[1, 0, 0], texture=texture)

    img = np.transpose(img, (1, 2, 0))
    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(shape_pred)
    plt.title('pred mesh')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(img_pred)
    plt.title('pred mesh w/texture')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(vp1)
    plt.title('different viewpoints')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(vp2)
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(vp3)
    plt.axis('off')
    plt.draw()
    #plt.show()
    plt.savefig("figure_" + opts.name + "__" + str(opts.num_train_epoch) + "__" + str(opts.use_train) + ".png")
    
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
#     ax = Axes3D(fig)#fig.add_subplot(111, projection='3d')
    pts = vert.cpu().numpy()
#     ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='r', marker='o')
    plt.scatter(pts[:,0], pts[:,1], c='r', marker='o')
    plt.xlim([-1, 2])
    plt.ylim([-1, 2])
    plt.savefig("3dfigure.png")
    
#    import ipdb
#    ipdb.set_trace()


def main(_):

    if opts.use_train >= 0:
        print("using train dataset")
        from .data import cub as cub_data
        
        data_module = cub_data
        dataloader = data_module.data_loader(opts, shuffle=False)
        for i, b in enumerate(dataloader):
            if i == opts.use_train:
                batch = b
                img = batch['img'].numpy().reshape([3, 224, 224])
                print("img shape", img.shape)
#         batch = dataloader[opts.use_train]
        
        
    else:
        img = preprocess_image(opts.img_path, img_size=opts.img_size)

        batch = {'img': torch.Tensor(np.expand_dims(img, 0))}

    predictor = pred_util.MeshPredictor(opts)
    outputs = predictor.predict(batch)

    # This is resolution
    renderer = predictor.vis_rend
    renderer.set_light_dir([0, 1, -1], 0.4)

    if opts.use_train >= 0:
        visualize(img, outputs, predictor.vis_rend)
    else:
        visualize(img, outputs, predictor.vis_rend)


if __name__ == '__main__':
    opts.batch_size = 1
    app.run(main)
