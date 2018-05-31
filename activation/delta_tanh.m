function result = delta_tanh(z)
	result = 1 - tanh(z).*tanh(z);
end