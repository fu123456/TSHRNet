% using iterative Otsu's algorithm to produce the specular highlight mask for a given input image

function mask=Rolling_L_Otsu(image,iter_num)
    [h,w,k]=size(image);
    if k==3
        Lab=rgb2lab(image);
    end
    if k==1
        Lab=image;
    end
    L=Lab(:,:,1);
    L=(L-min(L(:)))./(max(L(:))-min(L(:)));
    mask=ones(size(image,1),size(image,2));
    for i=1:iter_num
        L_temp=extract(L,mask);
        T=graythresh(L_temp);
        mask(find(L<=T))=0;
    end
end
