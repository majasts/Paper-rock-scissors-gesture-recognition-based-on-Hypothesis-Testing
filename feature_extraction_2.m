function num = feature_extraction_2(img)

s = size(img);
green_channel = img(:, :, 2); 
green_mask = green_channel > 0.75*max(green_channel(:));
num = sum(green_mask(:));
num = num/s(1)/s(2);
% figure() 
% imshow(green_mask)
end