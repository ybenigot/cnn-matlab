function image2=subsample(image1,factor)

	% let image1 be a square image whose dimension is divisible by factor
	offset = factor-1;
	dim = size(image1,1);
	j1=1;
	image2 = zeros(dim/factor,dim/factor);
	for i1=1:factor:dim
		j2=1;
		for i2=1:factor:dim
			%printf('i1 %d i2 %d j1 %d j2 %d\n',i1,i2,j1,j2);
			image2(j1,j2) = sum(image1(i1:i1+offset,i2:i2+offset)(:));%/(factor^2);
			j2 = j2 + 1;
		end%for
		j1 = j1 + 1;
	end%for

end%function