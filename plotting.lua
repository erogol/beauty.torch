require 'gnuplot'

local plotting = {}

function plotting.error_curve(stats, opt)
  local fn = paths.concat(opt.save,'train_error_curve.png')
  gnuplot.pngfigure(fn)
  gnuplot.title('Best Test Value : ' .. tostring(torch.Tensor(stats.testError):min()))
  local xs = torch.range(1, #stats.trainError)
  gnuplot.plot(
    { 'train_error', xs, torch.Tensor(stats.trainError), '-' },
    { 'test_error', xs, torch.Tensor(stats.testError), '-' }
  )
  gnuplot.axis({ 0, #stats.testError, 0, 100})
  gnuplot.xlabel('iteration')
  gnuplot.ylabel('error')
  gnuplot.plotflush()
end

function plotting.loss_curve(stats, opt)
  local fn = paths.concat(opt.save,'train_loss_curve.png')
  gnuplot.pngfigure(fn)
  gnuplot.title('Best Test Value : ' .. tostring(torch.Tensor(stats.testLoss):min()))
  local xs = torch.range(1, #stats.trainLoss)
  gnuplot.plot(
    { 'train_loss', xs, torch.Tensor(stats.trainLoss), '-' },
    { 'test_loss', xs, torch.Tensor(stats.testLoss), '-' }
  )
  gnuplot.axis({ 0, #stats.testLoss, 0, torch.Tensor(stats.testLoss):max()})
  gnuplot.xlabel('iteration')
  gnuplot.ylabel('loss')
  gnuplot.plotflush()
end

return plotting
