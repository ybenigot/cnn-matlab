function s=sumsq(t)
% emulation of octave sumsq for matlab

	u = t .* t;
	s =sum(u(:));

end%function