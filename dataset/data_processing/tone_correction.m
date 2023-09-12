% color correction for diffuse images in our dataset

data_dir='../../dataset/SSHR/test';
data_dir_subdirs=dir(data_dir);
subFolders=data_dir_subdirs([data_dir_subdirs.isdir]);
subFolders=subFolders(~ismember({subFolders(:).name},{'.','..'}));

% parameter
iter_num=1;

parfor i=1:numel(subFolders)
    dataDIR=fullfile(data_dir,subFolders(i).name);
    dataFiles=dir(fullfile(dataDIR,'*_i.jpg')); % find all input images
    for j=1:numel(dataFiles)
        [~,name,ext]=fileparts(fullfile(dataDIR,dataFiles(j).name));
        input_name=[name, ext];
        specular_residue_name=strrep(input_name,'_i.jpg','_r.jpg');
        diffuse_name=strrep(input_name,'_i.jpg','_d.jpg');
        diffuse_tc_name=strrep(input_name,'_i.jpg','_d_tc.jpg');
        input=imread(fullfile(dataDIR,dataFiles(j).name)); % input
        specular_residue=imread(fullfile(dataDIR,specular_residue_name)); % specular residue
        diffuse=imread(fullfile(dataDIR,diffuse_name)); % diffuse
        disp(input_name);

        % estimate mask using Otsu's method
        specular_highlight_mask=Rolling_L_Otsu(specular_residue,iter_num);
        specular_mask_name=strrep(input_name,'_i.jpg','_sm.jpg');
        specular_mask_name=[data_dir '/' subFolders(i).name '/' specular_mask_name];
        imwrite(specular_highlight_mask,specular_mask_name);

        % color transfer
        try
            % for checking the channels of images which could cause errors!
            [~,~,k1]=size(input);
            [~,~,k2]=size(specular_highlight_mask);
            [~,~,k3]=size(diffuse);
            if k3==1
                diffuse=repmat(diffuse,[1 1 3]);
            end
            if k1==1
                input=repmat(input,[1 1 3]);
            end

            diffuse_R = diffuse(:,:,1);
            diffuse_G = diffuse(:,:,2);
            diffuse_B = diffuse(:,:,3);
            input_R = input(:,:,1);
            input_G = input(:,:,2);
            input_B = input(:,:,3);

            num_non_highlight_pixels = sum(sum(1-specular_highlight_mask));

            index = 0;
            diffuse_r = zeros(num_non_highlight_pixels,1);
            diffuse_g = zeros(num_non_highlight_pixels,1);
            diffuse_b = zeros(num_non_highlight_pixels,1);
            input_r = zeros(num_non_highlight_pixels,1);
            input_g = zeros(num_non_highlight_pixels,1);
            input_b = zeros(num_non_highlight_pixels,1);

            for m=1:size(input_R,1)
                for n=1:size(input_R,2)
                    if specular_highlight_mask(m,n)==1
                        continue;
                    end
                    index = index+1;
                    diffuse_r(index,1) = diffuse_R(m,n);
                    diffuse_g(index,1) = diffuse_G(m,n);
                    diffuse_b(index,1) = diffuse_B(m,n);
                    input_r(index,1) = input_R(m,n);
                    input_g(index,1) = input_G(m,n);
                    input_b(index,1) = input_B(m,n);
                end
            end

            X_r = [ones(size(diffuse_r)) diffuse_r diffuse_g diffuse_b];
            M_r = regress(input_r,X_r);
            M_g = regress(input_g,X_r);
            M_b = regress(input_b,X_r);

            for m=1:size(diffuse_R,1)
                for n=1:size(diffuse_R,2)
                    x_gt = double([1 diffuse(m,n,1) diffuse(m,n,2) diffuse(m,n,3)]);
                    diffuse(m,n,1) = x_gt*M_r;
                    diffuse(m,n,2) = x_gt*M_g;
                    diffuse(m,n,3) = x_gt*M_b;
                end
            end

            diffuse_tc_name=[data_dir '/' subFolders(i).name '/' diffuse_tc_name];
            imwrite(diffuse, diffuse_tc_name);

        catch
            disp('With error!');
        end
    end
end
