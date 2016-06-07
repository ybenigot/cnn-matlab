function h=sigmoid(z)

	h = 1 ./ (1 + exp(-z));									% size layer_size x m

end%function
