local cURL = require 'cURL'

-- Url, location
local fileList = {
	{'https://www.adrianbulat.com/downloads/ECCV16/mpii_dataset.t7', 'dataset/mpii_dataset.t7'},
	{'https://www.adrianbulat.com/downloads/BinaryHumanPose/images/005808361.jpg', 'dataset/mpii/images/005808361.jpg'},
	{'https://www.adrianbulat.com/downloads/BinaryHumanPose/images/072245212.jpg', 'dataset/mpii/images/072245212.jpg'},
	{'https://www.adrianbulat.com/downloads/BinaryHumanPose/images/060754485.jpg', 'dataset/mpii/images/060754485.jpg'},
	{'https://www.adrianbulat.com/downloads/BinaryHumanPose/images/053710654.jpg', 'dataset/mpii/images/053710654.jpg'},
	{'https://www.adrianbulat.com/downloads/BinaryHumanPose/images/051074730.jpg', 'dataset/mpii/images/051074730.jpg'},
	{'https://www.adrianbulat.com/downloads/BinaryHumanPose/images/033761517.jpg', 'dataset/mpii/images/033761517.jpg'},
	{'https://www.adrianbulat.com/downloads/BinaryHumanPose/images/031800347.jpg', 'dataset/mpii/images/031800347.jpg'},
	{'https://www.adrianbulat.com/downloads/BinaryHumanPose/images/023724909.jpg', 'dataset/mpii/images/023724909.jpg'},
	{'https://www.adrianbulat.com/downloads/BinaryHumanPose/images/072818876.jpg', 'dataset/mpii/images/072818876.jpg'},
	{'https://www.adrianbulat.com/downloads/BinaryHumanPose/images/061062004.jpg', 'dataset/mpii/images/061062004.jpg'},
}

local m = cURL.multi()

for i = 1, #fileList do
	-- Open files
	fileList[i][2] = io.open(fileList[i][2], "w+b")

	-- Add the url handles
	fileList[i][1] = cURL.easy{url = fileList[i][1], writefunction = fileList[i][2]}
	m:add_handle(fileList[i][1])
end

print("Downloading files, please wait...")
-- Based on https://github.com/Lua-cURL/Lua-cURLv3/blob/master/examples/cURLv3/multi2.lua
local remain = #fileList
while remain > 0 do
	local last = m:perform()
	if last < remain then
		while true do
			local e, ok, err = m:info_read(true)
			if e == 0 then break end -- no more finished tasks
			if ok then
				print(e:getinfo_effective_url(), '-', '\027[00;92mOK\027[00m')
			else
				print(e:getinfo_effective_url(), '-', '\027[00;91mFail\027[00m')
			end
			e:close()
		end
	end 
	remain = last

	m:wait() 
end	
