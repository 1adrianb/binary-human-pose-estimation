require 'torch'
require 'nn'
require 'cudnn'
require 'paths'

require 'bnn'
require 'optim'

require 'draw'
local utils = require 'utils'
local opts = require('opts')(arg)

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local model = torch.load('models/humanpose_binary.t7')
model:evaluate()

local fileLists = utils.getFileList(opts)

local predictions = {}
for i = 1, #fileLists do
	local img = torch.load(fileLists[i].image)
	if opts.mode == 'demo' and opts.imagepath == '' then
		img = utils.crop(img, fileLists[i].center, fileLists[i].scale, 256)
	end
	img = img:cuda():view(1,3,256,256)
	
	local output = model:forward(img):add(utils.flip(utils.shuffleLR(model:forward(utils.flip(img)))))
	
	local preds_hm, preds_img = utils.getPreds(output, fileLists[i].center, fileLists[i].scale)
	
	if opts.mode == 'demo' then
		utils.plot(img,preds_hm)
		io.read() -- Wait for user input
	end
	
	if opts.mode == 'eval' then
		predictions[i] = preds_img:clone()
	end
end

if opts.mode == 'eval' then
	local dists = utils.calcDistance(predictions,fileLists)
	utils.calculateMetrics(dists)
end