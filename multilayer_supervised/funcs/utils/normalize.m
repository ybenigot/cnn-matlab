function [m,v] = normalize(data)

	fprintf('PREPROCESSING for zero mean and unit variance\n');

	m=mean(data(:));
	v=var(data(:));

	data=(data-m)/v;

end%function