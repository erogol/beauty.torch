--
-- Convert GPU model to CPU compatible version
-- Author: Eren Golge -  erengolge@gmail.com
--

require 'torch'
require 'cudnn'
require 'nn'
require 'cunn'
require "models/dropresnet"

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('convert GPU model to CPU version')
cmd:text()
cmd:text('Options')
cmd:option('-loadPath','','Path to load GPU model.')
cmd:option('-savePath','', 'Path to save CPU model')
cmd:text()

params = cmd:parse(arg)

print(params.loadPath)
local model = torch.load(params.loadPath)
model_cpu = cudnn.convert(model, nn):float()
torch.save(params.savePath, model_cpu)
