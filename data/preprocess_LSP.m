% create: labels(.txt), list(.txt)

clear

%set
lsp_foldername = 'lsp_dataset'; % 'lsp_dataset', 'lspet_dataset', 'lsp_dataset_original'
train_test = 'train'; % 'train', 'test'
check_labels = 0;
matrix_correspond = [1:19;10 11 12 4 5 6 0 9 8 7 3 2 1 0 14 13 0 0 0];
% keypoints LS  LE  LW  LH  LK  LA  LT  RS  RE  RW  RH  RK  RA  RT  HT  Ne  No  MS  MH
% ours      1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19
% LSP       10  11  12  4   5   6   0   9   8   7   3   2   1   0   14  13  0   0   0

m = find(matrix_correspond(2,:)~=0);
matrix_correspond = matrix_correspond(:,m);
% keypoints LS  LE  LW  LH  LK  LA  RS  RE  RW  RH  RK  RA  HT  Ne
% ours      1   2   3   4   5   6   8   9   10  11  12  13  15  16
% LSP       10  11  12  4   5   6   9   8   7   3   2   1   14  13
[~,numKeypoints_gt] = size(matrix_correspond);
load([lsp_foldername, '/joints.mat']);
if(strcmp(lsp_foldername, 'lspet_dataset'))
    joints = permute(joints, [2 1 3]);
elseif(strcmp(lsp_foldername, 'lsp_dataset') || strcmp(lsp_foldername, 'lsp_dataset_original'))
    joints(3,:,:) = abs(joints(3,:,:)-1); % visible:1, blocked:0
end

% labels
labels_foldername = [lsp_foldername, '/labels/labels'];
if ~exist(labels_foldername, 'dir')
  mkdir(labels_foldername);
end

% list
list_foldername = [lsp_foldername, '/list'];
if ~exist(list_foldername, 'dir')
  mkdir(list_foldername);
end
fileID = fopen([list_foldername, '/LSP_', train_test, '.txt'], 'w');
current_directory = cd;

num_imgs = 999;
complement = '4';
if(strcmp(train_test, 'train'))
    if(strcmp(lsp_foldername, 'lspet_dataset'))
        num_imgs = 9999;
        complement = '5';
    end
    ii = 1;
elseif(strcmp(train_test, 'test'))
    if(strcmp(lsp_foldername, 'lspet_dataset'))
        error('lspet_dataset should not be testing data.');
    end
    ii = 1001;
end

for i = ii:(ii+num_imgs)
    if(mod(i,100)==0)
        disp(i)
    end
    im_num = num2str(i, ['%0', complement, '.0f']);
    im_filename = ['im', im_num, '.jpg'];
    im_path = [lsp_foldername, '/images/', im_filename];
    im = imread(im_path);
    coords = joints(:,:,i);
    
    % adjust the order and the value (x,y,v,corresponding number)
    coords_adjusted = zeros(4,numKeypoints_gt);
    coords_adjusted(1:3,:) = coords(:,matrix_correspond(2,:));
    coords_adjusted(4,:) = matrix_correspond(1,:);
    
    % eliminate the unlabeled keypoints(lspet_dataset)
    if(strcmp(lsp_foldername, 'lspet_dataset'))
        coords_adjusted = coords_adjusted(:,logical(coords_adjusted(3,:)));
    end
    [~,numKeypoints] = size(coords_adjusted);
    
    % check the labels
    if(check_labels)
        imshow(im);
        hold on
        for j = 1:numKeypoints
            if(coords_adjusted(3,j))
                text(coords_adjusted(1,j), coords_adjusted(2,j), num2str(coords_adjusted(4,j)), 'Color', 'red', 'FontSize', 16);
            else
                text(coords_adjusted(1,j), coords_adjusted(2,j), num2str(coords_adjusted(4,j)), 'Color', 'blue', 'FontSize', 12);
            end
        end
        title(num2str(i))
        hold off
        pause();
    end
    
    % save the keypoints information
    fileID_labels = fopen([labels_foldername, '/', strrep(im_filename, 'jpg', 'txt')], 'w');
    for j = 1:numKeypoints
        fprintf(fileID_labels, '%d %.4f %.4f %d\n', coords_adjusted(4,j), coords_adjusted(1,j), coords_adjusted(2,j), coords_adjusted(3,j));
    end
    fclose(fileID_labels);
    
    % create list
    fprintf(fileID, [current_directory, '/', im_path, '\n']);
    
end
fclose(fileID);