local utils = {}

function utils.getTransform(center, scale, res)
	local h = 200*scale
	local t = torch.eye(3)
	
	-- Scale
	t[1][1] = res/h
	t[2][2] = res/h
	
	-- Translate
	t[1][3] = res*(-center[1]/h+0.5)
	t[2][3] = res*(-center[2]/h+0.5)

	return t
end

-- Transform the coordinates from the original image space to the cropped one
function utils.transform(pt, center, scale, res, invert)
    -- Define the transformation matrix
    local pt_new = torch.ones(3)
    pt_new[1], pt_new[2] = pt[1], pt[2]
    local t = utils.getTransform(center, scale, res)
    if invert then
        t = torch.inverse(t)
    end
    local new_point = (t*pt_new):sub(1,2):int()
    return new_point
end

-- Crop based on the image center & scale
function utils.crop(img, center, scale, res)
    local l1 = utils.transform({1,1}, center, scale, res, true)
    local l2 = utils.transform({res,res}, center, scale, res, true)

    local pad = math.floor(torch.norm((l1 - l2):float())/2 - (l2[1]-l1[1])/2)
    
    if img:size():size() < 3 then
      img = torch.repeatTensor(img,3,1,1)
    end

    local newDim = torch.IntTensor({img:size()[1], l2[2] - l1[2], l2[1] - l1[1]})
    local newImg = torch.zeros(newDim[1],newDim[2],newDim[3])
    local height, width = img:size()[2], img:size()[3]

    local newX = torch.Tensor({math.max(1, -l1[1]+1), math.min(l2[1], width) - l1[1]})
    local newY = torch.Tensor({math.max(1, -l1[2]+1), math.min(l2[2], height) - l1[2]})
    local oldX = torch.Tensor({math.max(1, l1[1]+1), math.min(l2[1], width)})
    local oldY = torch.Tensor({math.max(1, l1[2]+1), math.min(l2[2], height)})

    newImg:sub(1,newDim[1],newY[1],newY[2],newX[1],newX[2]):copy(img:sub(1,newDim[1],oldY[1],oldY[2],oldX[1],oldX[2]))

    newImg = image.scale(newImg,res,res)
    return newImg
end

function utils.getPreds(heatmaps, center, scale)
    if heatmaps:size():size() == 3 then heatmaps = heatmaps:view(1, unpack(heatmaps:size():totable())) end

    -- Get locations of maximum activations
    local max, idx = torch.max(heatmaps:view(heatmaps:size(1), heatmaps:size(2), heatmaps:size(3) * heatmaps:size(4)), 3)
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % heatmaps:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(heatmaps:size(3)):floor():add(1)

    for i = 1,preds:size(1) do        
        for j = 1,preds:size(2) do
            local hm = heatmaps[{i,j,{}}]
            local pX, pY = preds[{i,j,1}], preds[{i,j,2}]
            if pX > 1 and pX < 64 and pY > 1 and pY < 64 then
                local diff = torch.FloatTensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
                preds[i][j]:add(diff:sign():mul(.25))
            end
        end
    end
    preds:add(-0.5)

    -- Get the coordinates in the original space
    local preds_orig = torch.zeros(preds:size())
    for i = 1, heatmaps:size(1) do
        for j = 1, heatmaps:size(2) do
            preds_orig[i][j] = utils.transform(preds[i][j],center,scale,heatmaps:size(3),true)
        end
    end
    return preds, preds_orig
end

function utils.shuffleLR(x)
    local dim
    if x:nDimension() == 4 then
        dim = 2
    else
        assert(x:nDimension() == 3)
        dim = 1
    end

    local matched_parts = {
        {1,6},   {2,5},   {3,4},
        {11,16}, {12,15}, {13,14}
    }

    for i = 1,#matched_parts do
        local idx1, idx2 = unpack(matched_parts[i])
        local tmp = x:narrow(dim, idx1, 1):clone()
        x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
        x:narrow(dim, idx2, 1):copy(tmp)
    end

    return x
end

function utils.flip(x)
    local y = torch.FloatTensor(x:size())
    for i = 1, x:size(1) do
        image.hflip(y[i], x[i]:float())
    end
    return y:typeAs(x)
end

function utils.calcDistance(predictions,groundTruth)
  local n = predictions:size()[1]
  gnds = torch.Tensor(n,16,2)
  for i=1,n do
    gnds[{{i},{},{}}] = groundTruth[i].points
  end

  local dists = torch.Tensor(predictions:size(2),predictions:size(1))
  -- Calculate L2
	for i = 1,predictions:size(1) do
		for j = 1,predictions:size(2) do
			if gnds[i][j][1] > 1 and gnds[i][j][2] > 1 then
				dists[j][i] = torch.dist(gnds[i][j],predictions[i][j])/groundTruth[i].headSize
			else
				dists[j][i] = -1
			end
		end
	end

  return dists
end

function utils.getFileList(opts)
	local fileLists = {}

	if opts.imagepath ~= '' then
		fileLists[1] = {opts.imagepath}
	else
		fileLists = torch.load('dataset/mpii_dataset.t7')
        if opts.mode == 'demo' then
            local tempFileList = {}
            local idxs = {1,5,16,17,18,24,28,63,66,104}
            for i = 1, #idxs do
                tempFileList[i] = fileLists[idxs[i]]
            end
            fileLists = tempFileList
        end
	end
    print(fileLists)
	return fileLists
end

-- Requires qtlua
function utils.plot(surface, points)
	
    gnuplot.figure(1)
    gnuplot.raw("set size ratio -1")
    gnuplot.raw("unset key; unset tics; unset border;")
    gnuplot.raw("plot '"..surface.."' binary filetype=jpg with rgbimage")
   
end

function utils.calculateMetrics(dists)
    local threshold = 0.5
    dists:apply(function(x)
        if x>=0 and x<= threshold then 
            return 1
        elseif x>threshold then 
            return 0
        end
    end)

    local count = torch.zeros(16)
    local sums = torch.zeros(16)
    for i=1,16 do
        dists[i]:apply(function(x)
            if x ~= -1 then
                count[i] = count[i] + 1
                sums[i] = sums[i] + x
            end
        end)
    end

    local partNames = {'Head', 'Knee', 'Ankle', 'Shoulder', 'Elbow', 'Wrist', 'Hip'}
    local partsC =  torch.Tensor({{9,10},{2,5},{1,6},{13,14},{12,15},{11,16},{3,4}})
    print('PCKh results:')
    for i=1,#partNames do
        print(partNames[i]..': ',(sums[partsC[i][1]]/count[partsC[i][1]]+sums[partsC[i][2]]/count[partsC[i][1]])*100/2)
    end    
end

return utils