# mnist-dropout-lrdecay

Simple MNIST dataset degit recognization python code using Tensorflow.
Dropout & decayed learning-rate is used to increase accuracy.
I am using 4 hidden layers, ReLu activation (for hidden), softmax (for output).
Cross entropy to minimize errors, AdamOptimizer to optimize it.
<br/><br/>
Convolutional networks gives above 99% accuracy. For that we need to stack few Conv2D layers in Keras. <br/>
Dropout, BatchNormalization after each Conv layer also should be used to get accurate results.
