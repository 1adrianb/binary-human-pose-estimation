require 'image'
-- Implimentation of the Bresenham's Algorithm, can do antialising, disabled by default
function image.drawLine(surface, startxy, endxy, width, color)
	assert(surface:nDimension()==3, "The surface must have 3 dimensions.")

	local sx, sy = startxy[1]<endxy[1] and 1 or -1, startxy[2]<endxy[2] and 1 or -1
	local dist = torch.dist(endxy,startxy)
	local diff = torch.abs(startxy-endxy)
	local err = torch.sum(diff)

	local err2, x2, x2;
	width=(width+1)/2

	while true do
		surface[{{},startxy[1],startxy[2]}]:fill(color)
		err2 = err; x2 = startxy[1];
		if (2*err2 >= -diff[1]) then
			err2 = err2 + diff[2]
			y2 = startxy[2];
			while(err2<dist*width and (endxy[2]~=y2 or diff[1]>diff[2])) do
				y2 = y2+sy
				surface[{{},startxy[1],y2}]:fill(color)
				err2 = err2 + diff[1]
			end
			if (startxy[1] == endxy[1]) then break end
			err2 = err; err = err - diff[2]; startxy[1] = startxy[1] + sx
		end
		if (2*err2 <= diff[2]) then
			err2 = diff[1]-err2
			while (err2 < dist*width and (endxy[1]~=x2 or diff[1]<diff[2])) do
				x2 = x2+sx
				surface[{{},x2,startxy[2]}]:fill(color)
				err2 = err2 + diff[2]
			end
			if (startxy[2] == endxy[2]) then break end
			err = err + diff[1]; startxy[2] = startxy[2] + sy;
		end
	end

	return surface
end

-- Basic method to draw circles
function image.drawPoint(surface, point, r, color)
	for x = -r, r do
		local height = math.sqrt(r*r-x*x)
		
		for y = -height, height do
			surface[{{},x+point[1],y+point[2]}]:fill(color)
		end
	end
	return surface
end


