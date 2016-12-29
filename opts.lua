--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 ResNet Training script')
    cmd:text('See https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md for examples')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-data',       '',         'Path to dataset')
    cmd:option('-compute_mean_std', 'false',   'Compute mean and std')
    cmd:option('-dataset',    'imagenet', 'Options: imagenet | cifar10 | cifar100')
    cmd:option('-manualSeed', 0,          'Manually set RNG seed')
    cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
    cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
    cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
    cmd:option('-gen',        'gen',      'Path to save generated files')
    ------------- Data options ------------------------
    cmd:option('-nThreads',        2, 'number of data loading threads')
    ------------- Training options --------------------
    cmd:option('-nEpochs',           0,          'Number of total epochs to run')
    cmd:option('-epochNumber',       1,          'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',         32,         'mini-batch size (1 = pure stochastic)')
    cmd:option('-classWeighting',    'false',      'weight updates with reverse class frequency')
    --cmd:option('-uneLastLayer', 'false',      'scale last layer learning rate by 10')
    cmd:option('-testOnly',          'false',    'Run on validation set only')
    cmd:option('-tenCrop',           'false',    'Ten-crop testing')
    cmd:option('-resume',            'none',     'Path to directory containing checkpoint')
    cmd:option('-metric',            'accuracy', 'Metric to use in traning (f1_score, accuracy)')
    cmd:option('-save',              './',       'Path to save checkpoint models')
    cmd:option('-checkpoint',        'false',       'Save model after each epoch (true or false) (false)')
    cmd:option('-freeze',             'false',       'Freez conv layers for fine-tuning')
    cmd:option('-model_init_LR',     -1,         'Define a small LR to init model for 5 epochs. If it is below 0, ignored. (-1)')
    ---------- Optimization options ----------------------
    cmd:option('-LR',              0.1,   'initial learning rate')
    cmd:option('-momentum',        0.9,   'momentum')
    cmd:option('-weightDecay',     1e-4,  'weight decay')
    cmd:option('-LR_decay_step',   10,    'define number of steps to decay LR by 0.1')
    cmd:option('-optimizer',       'sgd',    'set th optimizer. sgd, adam (sgd)')
    ---------- Model options ----------------------------------
    cmd:option('-netType',      'resnet', 'Options: resnet | preresnet | dropresnet')
    cmd:option('-depth',        34,       'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
    cmd:option('-shortcutType', '',       'Options: A | B | C')
    cmd:option('-retrain',      'none',   'Path to model to retrain with')
    cmd:option('-optimState',   'none',   'Path to an optimState to reload from')
    ---------- Model options ----------------------------------
    cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
    cmd:option('-optnet',          'false', 'Use optnet to reduce memory usage')
    cmd:option('-resetClassifier', 'false', 'Reset the fully connected layer for fine-tuning')
    cmd:option('-nClasses',         0,      'Number of classes in the dataset')
    ---------- DropResNet options ----------------------------------
    cmd:option('-deathRate', 0.5, 'in stochastic layer resnet death rate of layers (0.5)')
    cmd:option('-deathMode', 'uniform', 'uniform or lin_decay (uniform)')
    cmd:text()

    local opt = cmd:parse(arg or {})

    opt.testOnly = opt.testOnly ~= 'false'
    opt.tenCrop = opt.tenCrop ~= 'false'
    opt.shareGradInput = opt.shareGradInput ~= 'false'
    opt.optnet = opt.optnet ~= 'false'
    opt.resetClassifier = opt.resetClassifier ~= 'false'
    opt.classWeighting = opt.classWeighting ~= 'false'
    opt.computeMeanStd = opt.computeMeanStd ~= 'false'
    -- opt.finetuneLastLayer = opt.finetuneLastLayer ~= 'false'
    opt.freeze = opt.freeze ~= 'false'

    -- set folder name to save model checkpoints
    -- add date/time
    opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''))
    print(" => Saving path is "..opt.save)

    if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
        cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
    end

    -- Handle the most common case of missing -data flag
    local trainDir = paths.concat(opt.data, 'train')
    if not paths.dirp(opt.data) then
        cmd:error('error: missing ImageNet data directory')
    elseif not paths.dirp(trainDir) then
        cmd:error('error: ImageNet missing `train` directory: ' .. trainDir)
    end
    -- Default shortcutType=B and nEpochs=90
    opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
    opt.nEpochs = opt.nEpochs == 0 and 90 or opt.nEpochs

    if opt.resetClassifier then
        if opt.nClasses == 0 then
            cmd:error('-nClasses required when resetClassifier is set')
        end
    end

    if opt.shareGradInput and opt.optnet then
        cmd:error('error: cannot use both -shareGradInput and -optnet')
    end

    -- set optimizer option
    print(" => ".. opt.optimizer .. ' optimizer is in use.')
    print(opt)

    -- save opt parameters to file
    local filename = paths.concat(opt.save,'opt.txt')
    local file = io.open(filename, 'w')
    for i,v in pairs(opt) do
        file:write(tostring(i)..' : '..tostring(v)..'\n')
    end
    file:close()
    torch.save(path.join(opt.save,'opt.t7'),opt)
    return opt
end

return M
