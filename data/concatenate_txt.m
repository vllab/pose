clear

% set
txt1_path = '/mnt/data/yfchen_data/Pose_Estimation/data/lsp_dataset/list/LSP_train.txt';
txt2_path = '/mnt/data/yfchen_data/Pose_Estimation/data/lspet_dataset/list/LSP_train.txt';

folder_txt_concat = 'txt_concat';
if ~exist(folder_txt_concat, 'dir')
  mkdir(folder_txt_concat);
end

% set the filename of the output file
version = 1;
txt_concat_path = [folder_txt_concat, '/concatenated_v1.txt'];
while (exist(txt_concat_path))
    version = version + 1;
    txt_concat_path = [folder_txt_concat, '/concatenated_v', num2str(version), '.txt'];
end

% read txt1
fileID1 = fopen(txt1_path, 'r');
im_paths1 = textscan(fileID1, '%s');
im_paths1 = im_paths1{1,1};
fclose(fileID1);

% read txt2
fileID2 = fopen(txt2_path, 'r');
im_paths2 = textscan(fileID2, '%s');
im_paths2 = im_paths2{1,1};
fclose(fileID2);

% concatenate
fileID_out = fopen(txt_concat_path, 'w');
for i = 1:numel(im_paths1)
    fprintf(fileID_out, '%s\n', im_paths1{i,1});
end
for i = 1:numel(im_paths2)
    fprintf(fileID_out, '%s\n', im_paths2{i,1});
end
fclose(fileID_out);

disp([cd, '/',txt_concat_path])