% extract the intensities of all ROI pixels specified by a mask

function imV=extract(im,mask)
    im=reshape(im,[],size(im,3));
    imV=im(find(mask),:);
end
