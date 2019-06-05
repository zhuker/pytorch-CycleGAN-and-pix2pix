import sys

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from data.single_dataset import SingleDataset
from models.pix2pix_model import Pix2PixModel
from options.base_options import BaseOptions
from options.test_options import TestOptions
from util import util

bo = BaseOptions()
opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
# opt.preprocess = ''
# opt.dataroot = './1024/'
# opt.preprocess = 'resize'
opt.preprocess = 'none'
# opt.dataroot = '/home/zhukov/ntsc/datasets/friends30deint/vhs/scene042/'
# opt.dataroot = './center/'
opt.dataroot = './center42/'

opt.num_threads = 0  # test code only supports num_threads = 1
opt.batch_size = 1  # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
opt.direction = 'AtoB'
opt.gpu_ids = [0]
opt.netG = 'unet_256'
opt.norm = 'batch'
opt.epoch = 15
opt.num_test = float("inf")

model = Pix2PixModel(opt)
model.save_dir = './vhs2film'
epoch = opt.epoch
model.load_networks(epoch)
model.eval()

dataset = SingleDataset(opt)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=1)

for i, data in enumerate(dataloader):
    # print(data)
    if i >= opt.num_test:  # only apply our model to opt.num_test images.
        break
    model.set_input(data)  # unpack data from data loader
    with torch.no_grad():
        model.forward()  # run inference
    image_numpy = util.tensor2im(model.fake_B)
    im = Image.fromarray(image_numpy)
    png_i = "out" + str(opt.epoch) + "/%06d.png" % i
    print(png_i)
    im.save(png_i)
