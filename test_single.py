import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
from util import util
from PIL import Image
import torchvision.transforms as transforms

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

model = create_model(opt)
visualizer = Visualizer(opt)
# test

dataset_path = opt.dataroot
dataset_output_path = opt.results_dir

transform_list  = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)

def process_file(filepath, outfilepath):
    A = Image.open(filepath).convert('RGB')
    A = transform(A)
    data = {'A': A.unsqueeze(0), 'A_paths': filepath }

    model.set_input(data)
    model.test()

    visuals = model.get_current_visuals()

    util.save_image(visuals['fake_A'], outfilepath)


try:
    os.mkdir(dataset_output_path)
except:
    pass

for i,filename in enumerate(os.listdir(dataset_path)):
    print('%04d: process image... %s' % (i, filename))
    filepath = os.path.join(dataset_path, filename)
    outfilepath = os.path.join(dataset_output_path, filename)

    process_file(filepath, outfilepath)

