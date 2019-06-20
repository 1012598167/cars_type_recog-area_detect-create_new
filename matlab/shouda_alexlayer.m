function [AlexLayer,opts]=alexnet()
inputLayer = imageInputLayer([227 227 3],'Name','Input');%指定图像大小227*227*3

middleLayers = [
convolution2dLayer([11 11], 96,'NumChannels',3,'Stride',4,'Name','conv1','Padding',0)
%过滤器的高度和宽度 过滤器数量（连接到同一输入区域的神经元数量 其决定了特征图的数量）
reluLayer('Name','relu1')
crossChannelNormalizationLayer(5,'Name','norm1')
maxPooling2dLayer(3, 'Stride', 2,'Name','pool1','Padding',0)

convolution2dLayer([5 5], 256, 'NumChannels',48,'Padding', 2,'Name','conv2','Stride',1)
reluLayer('Name','relu2')
crossChannelNormalizationLayer(5,'Name','norm2')
maxPooling2dLayer(3, 'Stride',2,'Name','pool2','Padding',0)

convolution2dLayer([3 3], 384, 'NumChannels',256,'Padding', 1,'Name','conv3','Stride',1)
reluLayer('Name','relu3')

convolution2dLayer([3 3], 384,'NumChannels',192, 'Padding', 1,'Name','conv4','Stride',1)
reluLayer('Name','relu4')

convolution2dLayer([3 3], 256, 'NumChannels',192,'Padding', 1,'Name','conv5','Stride',1)
reluLayer('Name','relu5')
maxPooling2dLayer(3, 'Stride',2,'Name','pool5','Padding',0)
];

finalLayers = [
fullyConnectedLayer(4096,'Name','fc6')
reluLayer('Name','relu6')
%caffe中有这一层
dropoutLayer(0.5,'Name','dropout6')

fullyConnectedLayer(4096,'Name','fc7')
reluLayer('Name','relu7')
%caffe中有这一层
dropoutLayer(0.5,'Name','dropout7')

%196种车
fullyConnectedLayer(196,'Name','fc8')
softmaxLayer('Name','softmax')
classificationLayer('Name','classification')
];

AlexLayer=[inputLayer
    middleLayers
    finalLayers];

opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'MaxEpochs', 500, ...
    'MiniBatchSize', 100, ...
    'Verbose', true);
end
