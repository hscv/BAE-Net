import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from modules.sample_generator import SampleGenerator
from modules.utils import crop_image2
from hsi_trans import X2Cube

class RegionDataset(data.Dataset):
    def __init__(self, img_list, gt, opts,chan=16):
        self.img_list = np.asarray(img_list)
        # print ('self.img_list = ',self.img_list)
        self.gt = np.array(gt)
        self.channel = chan

        self.batch_frames = opts['batch_frames']
        self.batch_pos = opts['batch_pos']
        self.batch_neg = opts['batch_neg']

        self.overlap_pos = opts['overlap_pos']
        self.overlap_neg = opts['overlap_neg']

        self.crop_size = opts['img_size']
        self.padding = opts['padding']

        self.flip = opts.get('flip', False)
        self.rotate = opts.get('rotate', 0)
        self.blur = opts.get('blur', 0)

        self.index = np.random.permutation(len(self.img_list))
        # self.index = np.arange(len(self.img_list))
        self.pointer = 0

        if self.channel == 16:
            image = Image.open(self.img_list[0]) #.convert('RGB')
            image = X2Cube(image) # numpy type
            imsize = image.shape[:2]
        else:
            # image = Image.open(self.img_list[0]).convert('RGB')
            # imsize = image.size
            image = Image.open(self.img_list[0]) #.convert('RGB')
            image = X2Cube(image) # numpy type
            # print ('image--res = ',image.shape)
            image = image[:,:,0:3]
            imsize = image.shape[:2]
            # print ('image--res111 = ',image.shape)

        self.pos_generator = SampleGenerator('uniform', imsize,
                opts['trans_pos'], opts['scale_pos'])
        # print ('self.pos_generator = ',self.pos_generator)
        self.neg_generator = SampleGenerator('uniform', imsize,
                opts['trans_neg'], opts['scale_neg'])
        # print ('self.neg_generator = ',self.neg_generator)

    def __iter__(self):
        return self

    def __next__(self):
        next_pointer = min(self.pointer + self.batch_frames, len(self.img_list))
        idx = self.index[self.pointer:next_pointer]
        if len(idx) < self.batch_frames:
            self.index = np.random.permutation(len(self.img_list))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer

        pos_regions = np.empty((0, self.channel, self.crop_size, self.crop_size), dtype='float32')
        neg_regions = np.empty((0, self.channel, self.crop_size, self.crop_size), dtype='float32')
        # print ('pos_regions.shape = ',pos_regions.shape)
        # print ('neg_regions.shape = ',neg_regions.shape)
        # print ('idx = ',idx)
        # return None,None
        # print ('self.img_list[idx] = ', self.img_list[idx])
        # print ('self.gt[idx] = ', self.gt[idx])
        for i, (img_path, bbox) in enumerate(zip(self.img_list[idx], self.gt[idx])):
            if self.channel == 16:
                image = Image.open(img_path)
                image = X2Cube(image)
                # print ('image.shape = ',image.shape)
            else:            
                image = Image.open(img_path).convert('RGB')
                image.thumbnail((512, 256))
                image = np.asarray(image)
                # image.thumbnail((512, 256))
                # print (image.shape)
            
            # print ('image.shape = ',image.shape)
            # print ('self.batch_frames = ',self.batch_frames)
            # print ('self.batch_pos = ',self.batch_pos,' , len(pos_regions) = ',len(pos_regions),' , self.batch_frames = ',self.batch_frames,' , i = ',i)
            # print ('self.batch_neg = ',self.batch_neg,' , len(neg_regions) = ',len(neg_regions),' , self.batch_frames = ',self.batch_frames,' , i = ',i)
            n_pos = (self.batch_pos - len(pos_regions)) // (self.batch_frames - i)
            n_neg = (self.batch_neg - len(neg_regions)) // (self.batch_frames - i)
            # print ('n_pos = ',n_pos,' , n_neg = ',n_neg)
            pos_examples = self.pos_generator(bbox, n_pos, overlap_range=self.overlap_pos)
            neg_examples = self.neg_generator(bbox, n_neg, overlap_range=self.overlap_neg)
            # print ('self.extract_regions(image, pos_examples).shape = ',self.extract_regions(image, pos_examples).shape)
            # print ('self.extract_regions(image, neg_generator).shape = ',self.extract_regions(image, neg_examples).shape)
            # print ('len(pos_examples) = ',len(pos_examples))
            pos_regions = np.concatenate((pos_regions, self.extract_regions(image, pos_examples)), axis=0)
            neg_regions = np.concatenate((neg_regions, self.extract_regions(image, neg_examples)), axis=0)

        pos_regions = torch.from_numpy(pos_regions)
        neg_regions = torch.from_numpy(neg_regions)
        # print ('--ll-- pos_regions.shape = ',pos_regions.shape)
        # print ('--ll-- neg_regions.shape = ',neg_regions.shape)
        return pos_regions, neg_regions

    next = __next__

    def extract_regions(self, image, samples):

        regions = np.zeros((len(samples), self.crop_size, self.crop_size, self.channel), dtype='uint8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image2(image, sample, self.crop_size, self.padding,
                    self.flip, self.rotate, self.blur)

        regions = regions.transpose(0, 3, 1, 2)
        regions = regions.astype('float32') - 128.
        return regions
