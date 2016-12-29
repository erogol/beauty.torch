--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--
require 'torch'
require 'nnlr'
require 'nn'

local optim = require 'optim'
local metrics = require 'utils/metrics'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
    self.model = model
    print(" => Learning rate " .. opt.LR)
    local initLearningRate = opt.LR
    -- set individual learning rates for params to freeze learning of conv layers
    local learningRates, weightDecays = nil, nil
    if opt.freeze then
        learningRates, weightDecays = self.model:getOptimConfig(opt.LR, opt.weightDecay)
        assert(learningRates:min() == 0)
        initLearningRate = 0
    end

    self.criterion = criterion
    self.lastLearningRate = 0
    self.optimState = optimState or {
        learningRates = learningRates,
        weightDecays = weightDecays,
        learningRate = initLearningRate,
        learningRateDecay = 0.0,
        momentum = opt.momentum,
        nesterov = true,
        dampening = 0.0,
        weightDecay = opt.weightDecay,
        -- Adam parameters
        beta1 = 0.9,
        beta2 = 0.999,
        -- LBFGS parameters
        -- maxIter = 5,
        -- lineSearch = optim.lswolfe
    }
    self.opt = opt
    self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
    -- Trains the model for a single epoch
    if self.opt.freeze then
        self.optimState.learningRates = self.optimState.learningRates * self:learningRatesDecay(epoch)
        if self.optimState.learningRates:max() ~= self.lastLearningRate then
            self.lastLearningRate = self.optimState.learningRates:max()
            print( " => Learning rate changed to .. ".. self.lastLearningRate)
        end
    else
        self.optimState.learningRate = self:learningRate(epoch)
        if self.optimState.learningRate ~= self.lastLearningRate then
            self.lastLearningRate = self.optimState.learningRate
            print( " => Learning rate changed to .. ".. self.lastLearningRate)
        end
    end

    local timer = torch.Timer()
    local dataTimer = torch.Timer()

    local function feval()
        return self.criterion.output, self.gradParams
    end

    local trainSize = dataloader:size()
    local lossSum = 0.0
    local N = 0

    if self.opt.netType == 'dropresnet' then
        addtables = {}
        for i=1,self.model:size() do
            if tostring(self.model:get(i)) == 'nn.ResidualDrop' then
                addtables[#addtables+1] = i
            end
        end

        ---- Sets the deathRate (1 - survival probability) for all residual blocks  ----
        for i,block in ipairs(addtables) do
            if self.opt.deathMode == 'uniform' then
                self.model:get(block).deathRate = self.opt.deathRate
            elseif self.opt.deathMode == 'lin_decay' then
                self.model:get(block).deathRate = i / #addtables * self.opt.deathRate
            else
                print('Invalid argument for deathMode!')
            end
        end
    end

    ---- Resets all gates to open ----
    function openAllGates()
        for i,block in ipairs(addtables) do self.model:get(block).gate = true end
    end

    print(' => Training epoch # ' .. epoch)
    -- set the batch norm to training mode
    self.model:training()
    for n, sample in dataloader:run() do

        -- Set drop rates and gates if dropresnet is active
        if self.opt.netType == 'dropresnet' then
            openAllGates()
            -- Randomly determines the gates to close, according to their survival probabilities
            for i,tb in ipairs(addtables) do
                if torch.rand(1)[1] < self.model:get(tb).deathRate then self.model:get(tb).gate = false end
            end
        end

        local dataTime = dataTimer:time().real

        -- Copy input and target to the GPU
        self:copyInputs(sample)
        local output = self.model:forward(self.input):float()
        local batchSize = output:size(1)
        local loss = self.criterion:forward(self.model.output, self.target)

        self.model:zeroGradParameters()
        self.criterion:backward(self.model.output, self.target)
        self.model:backward(self.input, self.criterion.gradInput)

        if self.opt.optimizer == 'adam' then
            optim.adam(feval, self.params, self.optimState)
        else
            optim.sgd(feval, self.params, self.optimState)
        end
        lossSum = lossSum + loss*batchSize
        N = N + batchSize

        print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f '):format(
        epoch, n, trainSize, timer:time().real, dataTime, loss))

        -- check that the storage didn't get changed do to an unfortunate getParameters call
        assert(self.params:storage() == self.model:parameters()[1]:storage())

        timer:reset()
        dataTimer:reset()
    end

    return lossSum / N
end

function Trainer:test(epoch, dataloader)

    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local size = dataloader:size()

    local nCrops = self.opt.tenCrop and 10 or 1
    local lossSum = 0.0, 0.0, 0.0, 0.0
    local N = 0


    self.model:evaluate()
    for n, sample in dataloader:run() do
        local dataTime = dataTimer:time().real

        -- Copy input and target to the GPU
        self:copyInputs(sample)

        local output = self.model:forward(self.input):float()
        local batchSize = output:size(1) / nCrops
        local loss = self.criterion:forward(self.model.output, self.target)

        lossSum = lossSum + loss*batchSize
        N = N + batchSize

        print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  loss %7.3f (%7.3f)'):format(
        epoch, n, size, timer:time().real, dataTime, loss, lossSum / N))

        timer:reset()
        dataTimer:reset()
    end
    self.model:training()

    print((' * Finished epoch # %d   loss: %7.3f \n'):format(
    epoch, lossSum / N))

    return lossSum / N
end

function Trainer:computeScore(output, target, nCrops)
    return  metrics.accuracy_score(output, target, nCrops)
end

function Trainer:copyInputs(sample)
    -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
    -- if using DataParallelTable. The target is always copied to a CUDA tensor
    self.input = self.input or (self.opt.nGPU == 1
    and torch.CudaTensor()
    or cutorch.createCudaHostTensor())
    self.target = self.target or torch.CudaTensor()

    self.input:resize(sample.input:size()):copy(sample.input)
    self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRatesDecay(epoch)
    -- Training schedule
    local decay = 0
    if self.opt.optimizer == 'adam' then
        decay = 1.0/math.sqrt(epoch)
    else
        decay = math.floor((epoch - 1) / self.opt.LR_decay_step)
        decay = math.pow(0.1, decay)
    end
    return decay
end


function Trainer:learningRate(epoch)
    -- Training schedule
    if self.opt.model_init_LR > 0 and epoch < 5 then
        return self.opt.model_init_LR
    elseif self.opt.optimizer == 'adam' then
        local decay = 0
        decay = 1.0/math.sqrt(epoch)
        print(' => Adam optimizer lr decay '.. decay)
        return self.opt.LR * decay
    else
        local decay = 0
        decay = math.floor((epoch - 1) / self.opt.LR_decay_step)
        return self.opt.LR * math.pow(0.1, decay)
    end
end

return M.Trainer
