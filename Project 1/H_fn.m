I = imread('dew on roses (blurred).tif');
subplot(2,2,1);imshow(I);title('Original Image'); 
H = fspecial('motion',0.00001,315);
MotionBlur = imfilter(I,H);
imwrite(MotionBlur,'Motion Blurred Image.tif');
subplot(2,2,2);imshow(MotionBlur);title('Motion Blurred Image');