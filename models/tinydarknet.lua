-- Based on model defined on http://pjreddie.com/darknet/tiny-darknet/

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = nn.ELU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
	local model = nn.Sequential()
	print(' | TinyDarknet is created')

	-- The SimpleNet model
	model:add(Convolution(3,16,3,3,1,1,1,1))
	model:add(ReLU())
	model:add(Max(2,2,2,2,0,0))

    model:add(Convolution(16,32,3,3,1,1,1,1))
	model:add(ReLU())
	model:add(Max(2,2,2,2,0,0))

    model:add(Convolution(32,16,1,1,1,1,0,0))
	model:add(ReLU())
    model:add(Convolution(16,128,3,3,1,1,1,1))
	model:add(ReLU())
    model:add(Convolution(128,16,1,1,1,1,0,0))
	model:add(ReLU())
    model:add(Convolution(16,128,3,3,1,1,1,1))
	model:add(ReLU())
	model:add(Max(2,2,2,2,0,0))

    model:add(Convolution(128,32,1,1,1,1,0,0))
	model:add(ReLU())
    model:add(Convolution(32,256,3,3,1,1,1,1))
	model:add(ReLU())
    model:add(Convolution(256,32,1,1,1,1,0,0))
	model:add(ReLU())
    model:add(Convolution(32,256,3,3,1,1,1,1))
	model:add(ReLU())
	model:add(Max(2,2,2,2,0,0))

    model:add(Convolution(256,64,1,1,1,1,0,0))
	model:add(ReLU())
    model:add(Convolution(64,512,3,3,1,1,1,1))
	model:add(ReLU())
    model:add(Convolution(512,64,1,1,1,1,0,0))
	model:add(ReLU())
    model:add(Convolution(64,512,3,3,1,1,1,1))
	model:add(ReLU())
    model:add(Convolution(512,128,1,1,1,1,0,0))
    model:add(ReLU())
    model:add(Convolution(128,1024,3,3,1,1,1,1))
    model:add(ReLU())
	model:add(Avg(14,14,1,1))

	model:add(nn.View(1024):setNumInputDims(3))
    model:add(nn.Dropout(0.5))
	model:add(nn.Linear(1024, 1))

	local function ConvInit(name)
		for k,v in pairs(model:findModules(name)) do
			local n = v.kW*v.kH*v.nOutputPlane
			v.weight:normal(0,math.sqrt(2/n))
			if cudnn.version >= 4000 then
				v.bias = nil
				v.gradBias = nil
			else
				v.bias:zero()
			end
		end
	end

	ConvInit('cudnn.SpatialConvolution')
	ConvInit('nn.SpatialConvolution')
	for k,v in pairs(model:findModules('nn.Linear')) do
		v.bias:zero()
	end
	model:cuda()

	if opt.cudnn == 'deterministic' then
		model:apply(function(m)
			if m.setMode then m:setMode(1,1,1) end
		end)
	end
	model:get(1).gradInput = nil

	return model
end

return createModel
