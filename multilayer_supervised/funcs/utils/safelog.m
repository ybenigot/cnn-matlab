function y=safelog(x)

	if sum((x<=0)(:))>0 
		x
		error('STOPS FOR INVALID LOG ARG');
	else
		y=log(x);
	end%if	

end%function