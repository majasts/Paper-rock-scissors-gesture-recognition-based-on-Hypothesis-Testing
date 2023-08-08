function num = feature_extraction_1(img)
    s = size(img);
    gray_img = rgb2gray(img);
    
    output_img = zeros(size(gray_img));
    max_p = max(max(gray_img));
    min_p = min(min(gray_img));
    adaptive_diff = (max_p-min_p)/10;

    for i = 1:size(gray_img, 1)
        for j = 1:size(gray_img, 2)
            if j > 1
                diff = abs(double(gray_img(i, j)) - double(gray_img(i, j-1)));
                if diff >  adaptive_diff %5
                    output_img(i, j) = 1;
                end
            end
            if i > 1
                diff = abs(double(gray_img(i, j)) - double(gray_img(i-1, j)));
                if diff >  adaptive_diff %150
                    output_img(i, j) = 1;
                end
            end
        end
    end
    % figure, imshow(output_img)
    num = sum(output_img(:));
    num = num/s(1)/s(2);
end
