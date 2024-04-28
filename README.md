# gogen

A simple manually built neural network model to generate names. The model has one input and one output layer with no hidden layers.

The model input is one-hot encoding of a char and the output is the probability score for each char. The chars include all lower-case letters plus '^' for the beginning of a names and '$' for the end of a name.

We use softmax to compute the probability scores. The loss function is the softmax cross-entropy.

Model paramaters is a matrix of 28 x 28 since the input is a vector of size 28 representing one-hot encoding of a char and the output is also of size 28 representing probability score of each char.
