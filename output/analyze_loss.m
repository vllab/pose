clear
% close all

loss_name = '1/weights_backup/loss.txt';
fileID = fopen(loss_name);
txt = textscan(fileID,'%d: %f, %f avg, %f rate');

fclose(fileID);

loss = txt{1,2};
[num,~] = size(loss);
figure(1);plot(txt{1,1}(1:num), loss(1:num)); title('loss');
figure(2);plot(txt{1,1}(1000:num), loss(1000:num)); title('loss(start from 1000)');
% figure();plot(loss_avg(20000:num));

loss_avg = txt{1,3};
figure(3);plot(txt{1,1}(1:num), loss_avg(1:num)); title('avg loss');
figure(4);plot(txt{1,1}(1000:num), loss_avg(1000:num)); title('avg loss(start from 1000)');
% figure();plot(loss_avg(20000:num));

figure(5);plot(txt{1,4}); title('learning rate');
