--
-- this is a simple data loader which pick the next instance by the random class
-- idx instead of random instance idx in the batch. This way, dta sampling is
-- balanced in case of imbalanced data distribution
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms_custom'
local ffi = require 'ffi'

local M = {}
local ImagenetDataset = torch.class('resnet.ImagenetDataset', M)

function ImagenetDataset:__init(imageInfo, opt, split)
    self.imageInfo = imageInfo[split]
    self.opt = opt
    self.split = split
    self.dir = paths.concat(opt.data, split)
    assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function ImagenetDataset:get(i)

    local nClasses = #self.imageInfo.classList
    local classIdx = torch.random(1, nClasses)
    local class = self.imageInfo.classList[classIdx]
    local imglist = self.imageInfo.imagePath[classIdx]
    local path = imglist[torch.random(1, #imglist)]

    local image = self:_loadImage(paths.concat(self.dir, path))

    return {
        input = image,
        imagepath = paths.concat(self.dir, path),
        target = classIdx,
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
    return self.imageInfo.nImages
end

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

function ImagenetDataset:preprocess()
    if self.split == 'train' then
        return t.Compose{
            t.Scale(256),
            t.RandomSizedCrop(224),
            -- t.ColorJitter({
            --   brightness = 0.1,
            --   contrast = 0.1,
            --    saturation = 0.1,
            -- }),
            t.Rotation(45),
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
