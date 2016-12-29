require "nn"
require "utils/NoBackprop"

local utils ={}
function utils.testSurgery(input, f, net, ...)
   local output1 = net:forward(input):clone()
   f(net,...)
   local output2 = net:forward(input):clone()
   print((output1 - output2):abs():max())
   assert((output1 - output2):abs():max() < 1e-5)
end

function utils.disableFeatureBackprop(features, maxLayer)
  local noBackpropModules = nn.Sequential()
  for i = 1,maxLayer do
    noBackpropModules:add(features.modules[1])
    features:remove(1)
  end
  features:insert(nn.NoBackprop(noBackpropModules):cuda(), 1)
end

return utils
