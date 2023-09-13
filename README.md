# mobileNet_practice

### Archetecture
https://miro.medium.com/v2/resize:fit:1400/format:webp/1*bqE59FvgpvoAQUMQ0WEoUA.png![image](https://github.com/uvaishnav/mobileNet_practice/assets/104910465/2a3513a4-cbe3-42bf-989f-3e1b7760c62f)




## DEFINING COMPONENTS OF MOBILENET ARCHETECTURE

*  __Why do we Use BatchNormalization()?__
  Batch normalization is a technique that helps to improve the training speed and performance of neural networks. 
  It does so by normalizing the inputs of each layer, which means that it adjusts the mean and variance of the inputs to be close to zero and one, respectively. 
  This reduces the internal covariate shift, which is the change in the distribution of inputs across different layers. 
  By reducing this shift, batch normalization makes the network more stable and less sensitive to the choice of hyperparameters, such as learning rate and weight initialization. 
  Batch normalization also has a regularizing effect, as it adds some noise to the inputs during training, which prevents overfitting.

*  __What happes in Expansion Phase ?__
  In the expansion phase, the input tensor is passed through a 1x1 convolution layer that increases the number of channels by a factor of expansion. 
  The expansion factor is usually 6, which means that the number of channels is multiplied by 6. For example, if the input tensor has 16 channels, the output tensor of the expansion phase will have 16 x 6 = 96 channels. 
  The purpose of the expansion phase is to increase the dimensionality of the input tensor before applying the depthwise convolution, which operates on each channel separately. 
  This way, the depthwise convolution can extract more features from the input tensor and improve the performance of the network. 

*  __What does " x=DepthwiseConv2D(kernel_size,strides=strides,padding='same')(x) " do?__
  A depthwise convolution is a type of convolution that operates on each input channel separately, using a different kernel for each channel. 
  The output of the depthwise convolution is a tensor with the same number of channels as the input, but each channel has been convolved with its own kernel. 
    

* __Why do we use this line,__
  __" if strides==(1,1) and inputs.shape[-1]==num_filters:__
        __return tf.keras.layers.Add()([inputs,x])__
    __return x__
  __" in PointWise Convolution ?__  
  To create a residual connection between the input and the output of the depthwise separable  convolution. 
  The residual connection is only added if the strides are (1, 1) and the number of input channels is equal to the number of output channels. 
  This is because the residual connection requires that the input and output have the same spatial dimensions and depth. 
  The residual connection helps to avoid the vanishing gradient problem and improve the performance of the network. 

  If the strides are (1, 1), it means that the spatial dimensions (height and width) of the input and output tensors are the same. 
  If inputs.shape[-1] == filters, it means that the depth (number of channels) of the input and output tensors are the same. 
  These two conditions are necessary for adding the tensors element-wise. If they are not met, the residual connection is skipped and only the output tensor is returned. 

* __Why is Relu6 not applied after PointWise Convolution ?__
  The reason why ReLU is not applied after the pointwise convolution in this case is because it would interfere with the residual connection. ReLU is a non-linear activation function that clips negative values to zero. If ReLU is applied before adding the input to the output, it would eliminate some information from the input and reduce the effectiveness of the residual connection.



