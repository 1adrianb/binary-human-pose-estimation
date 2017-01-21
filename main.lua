require 'torch'
require 'nn'
require 'cudnn'
require 'paths'

require 'bnn'
require 'optim'

require 'gnuplot'
require 'image'
require 'xlua'
local utils = require 'utils'
local opts = require('opts')(arg)

torch.setheaptracking(true)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local model = torch.load('models/humanpose_binary.t7')
model:evaluate()

local fileLists = utils.getFileList(opts)
local predictions = {}
local output = torch.CudaTensor(1,16,64,64)

if opts.mode == 'eval' then xlua.progress(0,#fileLists) end
for i = 1, #fileLists do
	fileLists[i].image = 'dataset/mpii/images/'..fileLists[i].image
	
	local img = image.load(fileLists[i].image)
	local originalSize = img:size()

	img = utils.crop(img, fileLists[i].center, fileLists[i].scale, 256)
	img = img:cuda():view(1,3,256,256)
	
	output:copy(model:forward(img))
	output:add(utils.flip(utils.shuffleLR(model:forward(utils.flip(img)))))

	local preds_hm, preds_img = utils.getPreds(output, fileLists[i].center, fileLists[i].scale)
	
	if opts.mode == 'demo' then
		utils.plot(fileLists[i].image,preds_img:view(16,2),torch.Tensor{originalSize[3],originalSize[2]})
		io.read() -- Wait for user input
	end
	
	if opts.mode == 'eval' then
		predictions[i] = preds_img:clone()
		xlua.progress(i, #fileLists)
	end
end

if opts.mode == 'demo' then gnuplot.closeall() end

if opts.mode == 'eval' then
	predictions = torch.cat(predictions,1)
	local dists = utils.calcDistance(predictions,fileLists)
	utils.calculateMetrics(dists)
end