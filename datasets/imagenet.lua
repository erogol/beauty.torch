--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local ImagenetDataset = torch.class('resnet.ImagenetDataset', M)

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function ImagenetDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = paths.concat(opt.data, split)
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

--- Use this function when you need new mean and std for your custom dataset
function ImagenetDataset:computeMeanStd()
    local tm = torch.Timer()
    local nSamples = 10000
    print(' => Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
    local meanEstimate = {0,0,0}
    for i=1,nSamples do
       local img = self:get(i)['input']
       for j=1,3 do
          meanEstimate[j] = meanEstimate[j] + img[j]:mean()
       end
    end
    for j=1,3 do
       meanEstimate[j] = meanEstimate[j] / nSamples
    end
    mean = meanEstimate

    print(' => Estimating the std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
    local stdEstimate = {0,0,0}
    for i=1,nSamples do
       local img = self:get(i)['input']
       for j=1,3 do
          stdEstimate[j] = stdEstimate[j] + img[j]:std()
       end
    end
    for j=1,3 do
       stdEstimate[j] = stdEstimate[j] / nSamples
    end
    std = stdEstimate

    meanstd.mean = mean
    meanstd.std = std
    print(" | => Mean: ")
    print(mean)
    print(" | => Std: ")
    print(std)
end

function ImagenetDataset:get(i)
   local path = ffi.string(self.imageInfo.imagePath[i]:data())

   local image = self:_loadImage(paths.concat(self.dir, path))
   local class = self.imageInfo.imageClass[i]

   return {
      input = image,
      imagepath = paths.concat(self.dir, path),
      target = class,
   }
end

function ImagenetDataset:_loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float')
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      print(tostring(path))
      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))

      input = image.decompress(b, 3, 'float')
   end

   return input
end

function ImagenetDataset:size()
   return self.imageInfo.imageClass:size(1)
end

function ImagenetDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.Scale(256),
         t.RandomSizedCrop(224),
        t.ColorJitter({
          brightness = 0.1,
          contrast = 0.1,
           saturation = 0.1,
        }),
         t.Rotation(30),
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
      }
   elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      return t.Compose{
         t.Scale(224),
         t.ColorNormalize(meanstd),
         Crop(224),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.ImagenetDataset
