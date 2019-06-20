aclass=[annotations.class];
afname={annotations.fname};
%取后两千个进行测试
true=0;
all=2000;
for n = 6001:(6001+all)
    TypeNum=test(['cars_train_nocrop/'  afname{n}],AlexNet_New,class_names);
    cstr=cellstr(TypeNum(1));
    strnum=cstr{1};
    numb=str2num(strnum);
    if (numb==aclass(n))
        true=true+1;
    end
end
disp('correct rate');
disp(true/all);
function [TypeNum]=test(fileRoad,AlexNet_New,class_names)
testImage=imread(fileRoad);
%disp(testImage.Labels);
testImage_=imresize(testImage,[227 227]);
TypeNum=classify(AlexNet_New,testImage_);
%TypeName=class_names(TypeNum);
%disp(TypeName);
%figure;%create a figure window
%imshow(testImage);
end