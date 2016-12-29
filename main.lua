--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'torchx'
require 'paths'
require 'optim'
require 'nn'
local plotting = require 'plotting'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

-- Logger
logger = optim.Logger(paths.concat(opt.save,'training.log'))
logger:setNames{"Training Error", 'Validation Error', "Training Loss", "Validation Loss"}

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

trainingStats = { testLoss={}, trainLoss={}, testError={}, trainError={}}

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge
local bestLoss = math.huge
local bestEpoch = math.huge
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testLoss = trainer:test(epoch, valLoader)

   -- Update training stats
   table.insert(trainingStats.trainLoss, trainLoss)
   table.insert(trainingStats.testLoss, testLoss)

   -- Update logger
   logger:add{trainLoss, testLoss}

   -- Plot learning curves
   plotting.loss_curve(trainingStats, opt)

   local bestModel = false
   if testLoss < bestLoss then
      bestModel = true
      bestLoss = testLoss
      bestEpoch = epoch
      print(string.format(' * Best Model -- epoch:%i  loss: %6.3f', bestEpoch, bestLoss))

   end

   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
end

print(string.format(' * Best Model -- epoch:%i  loss: %6.3f', bestEpoch, bestLoss))
