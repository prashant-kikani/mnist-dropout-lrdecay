# mnist-dropout-lrdecay

To classify handwritten numbers (from 0 to 9 individually) using MNIST dataset in Tensorflow Python.<br/>
Dropout & decayed learning-rate is used to increase accuracy.<br/>
I am using 4 hidden layers, ReLu activation (for hidden), softmax (for output).<br/>
Cross entropy to minimize errors, AdamOptimizer to optimize it.
<br/><br/>
Convolutional networks gives above 99% accuracy. For that we need to stack few Conv2D layers in Keras. <br/>
Dropout, BatchNormalization after each Conv layer also should be used to get accurate results.<br/>
<b>Image Augmentation</b> also increases the accuracy.
