%%
[AlexLayer_New , optionsTransfer]=FineTune(AlexNet)

%%
traindata=imageDatastore('cars_train_croped(227_227)/','LabelSource','none')
%%
%load 'cars_meta.mat';
%annotations.class即为标注信息
ac=[annotations.class]
traindata.Labels=categorical(ac(1:6000))%取前6000个进行测试
%traindata.Labels=categorical([class_names])
%%
unique(traindata.Labels)%看数量是不是196个
%%
AlexNet_New=trainNetwork(traindata,AlexLayer_New,optionsTransfer)


%%
function [AlexLayer_New , optionsTransfer]=FineTune(AlexNet)%改最后三层
AlexNet_reduce = AlexNet.Layers(1:end-3);
%add
Last3Layers = [
fullyConnectedLayer(196,'Name','fc8','WeightLearnRateFactor',10, 'BiasLearnRateFactor',20)
softmaxLayer('Name','softmax')
classificationLayer('Name','classification')
];
AlexLayer_New=[AlexNet_reduce
    Last3Layers];

optionsTransfer = trainingOptions('sgdm',...%有动量的随机梯度下降
         'MaxEpochs',10,...
         'InitialLearnRate',0.0005,...
         'Verbose',true,'MiniBatchSize', 100);%MiniBatchSize根据显卡内存而定
end
%%
%%
function test(fileRoad,AlexNet_New,class_names)#封装测试
testImage=imread(fileRoad);
testImage_=imresize(testImage,[227 227]);
TypeNum=classify(AlexNet_New,testImage_);
TypeName=class_names(TypeNum);
disp(TypeName);
figure;
imshow(testImage);
end
%%