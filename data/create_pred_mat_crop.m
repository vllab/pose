% use the output of c code to generate the preds .mat file
clear

num_model = '1';
iteration = '25000';
version = '1';

lsp_foldername = 'lsp_dataset_original/';
crop_foldername = [lsp_foldername, 'crop/crop_v', version,'/'];
bbox_foldername = [lsp_foldername, 'crop/bbox/'];
bbox = load([bbox_foldername, 'bbox_v', version], 'bbox'); bbox = bbox.bbox;

netout_foldername = ['/mnt/data/yfchen_data/Pose_Estimation/output/', num_model, '/network_output/'];
pred_mat_foldername = ['/mnt/data/yfchen_data/Pose_Estimation/output/', num_model, '/preds_mat/'];

threshold = 0.005;
side = 14;
locations = side*side;
classes = 20;
numpred = 4;
coords = 2;
show_pred = 1;

% keypoints_label = {'LeftShoulder', 'LeftElbow', 'LeftWrist', 'LeftHip', ...
%                    'LeftKnee', 'LeftAnkle', 'LeftToes', 'RightShoulder', ...
%                    'LeftElbow', 'RightWrist', 'RightHip', 'RightKnee', ...
%                    'RightAnkle', 'RightToes', 'HeadTop', 'Neck', 'Nose', ...
%                    'MeanShoulder', 'MeanHip', 'FrontBack'};
keypoints_label = {'LS', 'LE', 'LW', 'LH', 'LK', 'LA', 'LT', ...
                   'RS', 'RE', 'RW', 'RH', 'RK', 'RA', 'RT', ...
                   'HT', 'Ne', 'No', 'MS', 'MH', 'FB'};
pred = zeros(2,14,1000);

for i = 1:1000
    if(mod(i,100)==0)
        disp(i)
    end
    im_name = ['im', num2str(1000+i, '%04.0f'), '.jpg'];
    im = imread([crop_foldername, im_name]);
    [h,w,~] = size(im);
    
    txt_name = ['im', num2str(1000+i, '%04.0f'), '.txt'];
    txt = textread([netout_foldername, txt_name]);
    
    % scores matrix
    class_potential = txt(1:locations,1:classes-1);
    confidence = txt(1+locations:2*locations,1:numpred);
    
    class_potential_rep = repmat(class_potential,1,numpred);
    class_potential_rep = reshape(class_potential_rep',classes-1,numpred*locations)';
    
    confidence_rep = reshape(confidence',1,locations*numpred)';
    confidence_rep = repmat(confidence_rep,1,classes-1);
    
    scores = confidence_rep.*class_potential_rep; % (side*side*numpred)*(classes-1)
    
    % coordinate
    coord = txt(1+2*locations:3*locations,1:numpred*coords);
    coords_reshape = reshape(coord',coords,locations*numpred)';
    
    % pick the keypoints whose scores are bigger than threshold
    scores_thresh_logical = scores>threshold;
    scores_thresh = scores.*(scores_thresh_logical);
    num_scores_thresh = sum(sum(scores_thresh_logical));
    [socres_thresh_sorted,IndexMatrix] = sort( reshape(scores_thresh, locations*numpred*(classes-1), 1), 'descend');
    
    socres_thresh_sorted = socres_thresh_sorted(1:num_scores_thresh,1);
    IndexMatrix = IndexMatrix(1:num_scores_thresh,1);
    class_belong = floor((IndexMatrix-1)/(locations*numpred))+1; % 1~classes-1
    location_belong = mod(IndexMatrix-1, locations*numpred); % 0~(side*side*numpred)-1
    
    % pick the predictions from the highest score
    keypoints_chosen_class = zeros(classes-1,5); % chosen, x, y, v, score
    keypoints_chosen_locations = zeros(locations*numpred,1);
    for j = 1:num_scores_thresh
        class = class_belong(j,1);
        location = location_belong(j,1);
        if(keypoints_chosen_class(class,1) == 0 && keypoints_chosen_locations(location+1,1) == 0)
            keypoints_chosen_class(class,1) = 1;
            keypoints_chosen_locations(location+1,1) = 1;
            row = floor(location/(side*numpred));
            col = floor(mod(location,(side*numpred))/numpred);
            keypoints_chosen_class(class,2) = w*( (col+coords_reshape(location+1,1))/side );
            keypoints_chosen_class(class,3) = h*( (row+coords_reshape(location+1,2))/side );
%             keypoints_chosen_class(class,4) = coords_reshape(location+1,coords);
            keypoints_chosen_class(class,5) = socres_thresh_sorted(j,1);
        end
    end
    
    % draw the predictions
    if(show_pred)
        imshow(im);
        hold on
        for j = 1:classes-1
            if(keypoints_chosen_class(j,1))
                if(mod(j,2) == 1)
                    label_location = 5;
                else
                    label_location = -5;
                end
                if( round(keypoints_chosen_class(j,4)) == 1)
                    plot(keypoints_chosen_class(j,2), keypoints_chosen_class(j,3), 'r.', 'MarkerSize', 10);
                    text(keypoints_chosen_class(j,2)+5, keypoints_chosen_class(j,3)+label_location, keypoints_label{1,j}, 'Color', 'red', 'FontSize', 14);
                else
                    plot(keypoints_chosen_class(j,2), keypoints_chosen_class(j,3), 'r.', 'MarkerSize', 10);
                    text(keypoints_chosen_class(j,2)+5, keypoints_chosen_class(j,3)+label_location, keypoints_label{1,j}, 'Color', 'yellow', 'FontSize', 16);
                end
            end
    %         pause();
        end
        hold off
        pause();
    end
    
    % adjust the order
    pred(:,:,i) = keypoints_chosen_class([13 12 11 4 5 6 10 9 8 1 2 3 16 15], 2:3)' + repmat(bbox(1:2, i),1,14);
end

% store the result
if(exist('iteration', 'var'))
    save([pred_mat_foldername, 'pred_lsp_pc_', iteration,'_v', version, '.mat'], 'pred');
else
    save([pred_mat_foldername, 'pred_lsp_pc_v', version, '.mat'], 'pred');
end


% 1 LS
% 2 LE
% 3 LW
% 4 LH
% 5 LK
% 6 LA
% 7 LT
% 
% 8 RS
% 9 RE
% 10 RW
% 11 RH
% 12 RK
% 13 RA
% 14 RT
% 
% 15 HT
% 16 Neck
% 17 Nose
% 
% 18 MS
% 19 MH
% 
% 20 FB