function testnan(x,s)

	if sum(isnan(x)(:))>0
		printf('%d NaN of %d - %s\n',sum(isnan(x)(:)),size(x(:),1),s);
		size(x)
		%error('NaN detected');
	end

end%function