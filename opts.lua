local function parse(arg)
	local cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Binary Human Pose demo script')
	cmd:text('Please visit https://www.adrianbulat.com for additional details')
	cmd:text()
	cmd:text('Options:')
	
	cmd:option('-imagepath',	'', 	'Path to the pre-processed(!) image')
	cmd:option('-mode',			'demo', 'Options: demo | eval')
	
	cmd:text()
	
	local opt = cmd:parse(arg or {})
	
	return opt 
end

return parse