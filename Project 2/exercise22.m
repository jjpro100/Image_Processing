

% MARR_HILDRETH.m - Marr-Hildreth operator example
% 
% This code implements Marr-Hildreth operator. It uses the second
% derivative of gaussian to create the template to convolute later, and
% finally, it detects zero-crossings to establish whether an edge was
% found or not.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com

clear;clc;

% Whether we preserve the real size of the image or we cut it.
% When the image is convoluted with the template, the borders are always
% lost.
preserve = 1;


% Read the image in gray scale
%im = imread('Fig1022(a)(building_original).tif');



im = imread('airplane in the sky.tif');

%g = imgaussfilt(im,4);
%laplacian = ones(3);
%laplacian(2,2) = -8;
g = fspecial('log',5,4);
c_LoG = conv2(im,g);

%thresh1 = 0;
%thresh2 = 0.04;

smim = c_LoG;
X = max(smim(:));
[rr,cc]=size(smim);
zc=zeros([rr,cc]);

for i=2:rr-1
    for j=2:cc-1
        if (smim(i,j)>0)
             if (smim(i,j+1)>=0 && smim(i,j-1)<0) || (smim(i,j+1)<0 && smim(i,j-1)>=0)
                             
                zc(i,j)= smim(i,j+1);
                        
            elseif (smim(i+1,j)>=0 && smim(i-1,j)<0) || (smim(i+1,j)<0 && smim(i-1,j)>=0)
                    zc(i,j)= smim(i,j+1);
            elseif (smim(i+1,j+1)>=0 && smim(i-1,j-1)<0) || (smim(i+1,j+1)<0 && smim(i-1,j-1)>=0)
                  zc(i,j)= smim(i,j+1);
            elseif (smim(i-1,j+1)>=0 && smim(i+1,j-1)<0) || (smim(i-1,j+1)<0 && smim(i+1,j-1)>=0)
                  zc(i,j)=smim(i,j+1);
            end
                        
        end
            
    end
end
%otpt=im2uint8(zc);
otpt=zc;
% thresholding
result= otpt>(X*0.04);

figure(1);
imshow(result)

rotI = imrotate(im,0,'crop');

%[H,T,R] = hough(result,'RhoResolution',5,'Theta',-90:2:89);
[H,T,R] = hough(result,'RhoResolution',5,'Theta',-90:2:88);
subplot(2,1,1);
imshow(result);
title('plane');
subplot(2,1,2);
imshow(imadjust(rescale(H)),'XData',T,'YData',R,'InitialMagnification','fit');
title('Hough.png');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
colormap(gca,hot);

P  = houghpeaks(H,20, 'threshold',ceil(0.20*max(H(:))));
x = T(P(:,2)); y = R(P(:,1));
plot(x,y,'o','color','blue');


lines = houghlines(result,T,R,P,'FillGap',5,'MinLength',7);
figure(6), imshow(rotI), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end

[rr,cc]=size(im);
img=zeros(rr,cc,3,'uint8');
%[rr,cc]=size(im);
%i=zeros([rr,cc]);
lines = houghlines(result,T,R,P,'FillGap',5,'MinLength',7);
figure(7), imshow(img), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end


im = imread('dew on roses.tif');
thresh = multithresh(im,2);
seg_I = imquantize(im,thresh);
RGB = label2rgb(seg_I);
figure(5);
imshow(RGB);
axis off
title('RGB Segmented Image');


% IDX = otsu(im,3);
% imagesc(IDX), axis image off 
% title('3 segments','FontWeight','bold') 



% g = fspecial('gaussian',5,4);
% 
% %g= [0 0 1 0 0;
% %        0 1 2 1 0;
% %        1 2 -16 2 1;
% %        0 1 2 1 0;
% %        0 0 1 0 0];
% 
% laplacian = ones(3);
% laplacian(2,2) = -8;
% 
% c_Gauss = conv2(im,g);
% c_LoG = conv2(c_Gauss,laplacian);
% 
% %c_LoG = conv2(im,g);
% %figure(5)
% figure(3);
% imshow(c_LoG,[])
% 
% %result = findZeroCrossings(c_LoG, preserve)
% 
% %thresh = 0.04;
% %result = edge(c_LoG,'zerocross',thresh, 4);
% 
% 
% smim = c_LoG;
% X = max(smim(:));
% [rr,cc]=size(smim);
% zc=zeros([rr,cc]);
% for i=2:rr-1
%     for j=2:cc-1
%         if (smim(i,j)>0)
%              if (smim(i,j+1)>=0 && smim(i,j-1)<0) || (smim(i,j+1)<0 && smim(i,j-1)>=0)
%                              
%                 zc(i,j)= smim(i,j+1);
%                         
%             elseif (smim(i+1,j)>=0 && smim(i-1,j)<0) || (smim(i+1,j)<0 && smim(i-1,j)>=0)
%                     zc(i,j)= smim(i,j+1);
%             elseif (smim(i+1,j+1)>=0 && smim(i-1,j-1)<0) || (smim(i+1,j+1)<0 && smim(i-1,j-1)>=0)
%                   zc(i,j)= smim(i,j+1);
%             elseif (smim(i-1,j+1)>=0 && smim(i+1,j-1)<0) || (smim(i-1,j+1)<0 && smim(i+1,j-1)>=0)
%                   zc(i,j)=smim(i,j+1);
%             end
%                         
%         end
%             
%     end
% end
% %otpt=im2uint8(zc);
% otpt=zc;
% % thresholding
% result= otpt>(X*0.04);
% 
% figure(2);
% imshow(result)
% % Calculate a matrix which will be the template to convolute. Sigma and
% % size of the kernel are given to the function.
% %%gauss = secDerGauss(1,5);
% 
% % Normalization
% % The trials I carried on showed me that there was no difference between
% % normalizing and not.
% %ratio = 1/(sum(sum(gauss)));
% %gauss = gauss.*ratio;
% 
% % Convolution through the whole image
% %%convoluted = myConv( im, gauss, preserve);
% 
% % Find zero-crossings
% %%result = findZeroCrossings(convoluted, preserve);
% 
% %%imshow(result);
% figure(1);
% %I = imread('airplane in the sky.tif');
% %I  = rgb2gray(RGB);
% %BW = edge(result,'canny');
% [H,T,R] = hough(result,'RhoResolution',5,'Theta',-90:0.5:89);
% subplot(2,1,1);
% imshow(result);
% title('plane');
% subplot(2,1,2);
% imshow(imadjust(rescale(H)),'XData',T,'YData',R,'InitialMagnification','fit');
% title('Hough.png');
% xlabel('\theta'), ylabel('\rho');
% axis on, axis normal, hold on;
% colormap(gca,hot);