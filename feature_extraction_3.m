function num = feature_extraction_3(img)
s = size(img);
img = rgb2gray(img);
mask = img <= 0.75*max(img);
num=sum(mask(:));
num = num/s(1)/s(2);
% figure, imshow(mask)
end