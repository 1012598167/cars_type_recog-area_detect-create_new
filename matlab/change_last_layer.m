%%
[AlexLayer_New , optionsTransfer]=FineTune(AlexNet)

%%
traindata=imageDatastore('cars_train_croped(227_227)/','LabelSource','none')
%%
%load 'cars_meta.mat';
%annotations.class��Ϊ��ע��Ϣ
ac=[annotations.class]
traindata.Labels=categorical(ac(1:6000))%ȡǰ6000�����в���
%traindata.Labels=categorical([class_names])
%%
unique(traindata.Labels)%�������ǲ���196��
%%
AlexNet_New=trainNetwork(traindata,AlexLayer_New,optionsTransfer)


%%
function [AlexLayer_New , optionsTransfer]=FineTune(AlexNet)%���������
AlexNet_reduce = AlexNet.Layers(1:end-3);
%add
Last3Layers = [
fullyConnectedLayer(196,'Name','fc8','WeightLearnRateFactor',10, 'BiasLearnRateFactor',20)
softmaxLayer('Name','softmax')
classificationLayer('Name','classification')
];
AlexLayer_New=[AlexNet_reduce
    Last3Layers];

optionsTransfer = trainingOptions('sgdm',...%�ж���������ݶ��½�
         'MaxEpochs',10,...
         'InitialLearnRate',0.0005,...
         'Verbose',true,'MiniBatchSize', 100);%MiniBatchSize�����Կ��ڴ����
end
%%
%%
function test(fileRoad,AlexNet_New,class_names)#��װ����
testImage=imread(fileRoad);
testImage_=imresize(testImage,[227 227]);
TypeNum=classify(AlexNet_New,testImage_);
TypeName=class_names(TypeNum);
disp(TypeName);
figure;
imshow(testImage);
end
%%