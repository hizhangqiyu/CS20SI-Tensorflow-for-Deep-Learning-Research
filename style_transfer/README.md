# Style transfer
Assignment of style transfer.
Please refert to the paper for [style transfer](https://arxiv.org/pdf/1508.06576v2.pdf) and paper for [vgg model](https://arxiv.org/pdf/1409.1556.pdf).

## Notice
* Weights and biases loaded from pre-trained VGG model should be constant type. The trainable vabiables are pixel values.
* Tensorflow use NHWC by default, but no batch dim for style transfer, paper ignore it.
