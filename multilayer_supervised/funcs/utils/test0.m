function test0(D,s)

% triggers an exception if the argument tensor is full of zeroes

	a=sum((D==0)(:));
	b=sum((D~=0)(:));
	if (b==0)
		fprintf('%s: zeros %d, non zeroes %d\n',s,a,b);
		%fflush(stdout);
	end%if	

end%function