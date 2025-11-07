clc;
clear;
close all;

% 定义输入文件夹和输出文件夹
input_dir = 'D:\桌面文件\Brightness_reduction\dataset\LOLv1\low';  % 替换为你的输入文件夹路径，例如 'C:\images\input'
output_dir = 'D:\桌面文件\PNAR-master\output';  % 替换为你的输出文件夹路径，例如 'C:\images\output'

% 如果输出文件夹不存在，则创建
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 获取输入文件夹中的所有图像文件（假设处理PNG、JPG、JPEG文件）
file_list = dir(fullfile(input_dir, '*.png'));
file_list = [file_list; dir(fullfile(input_dir, '*.jpg'))];
file_list = [file_list; dir(fullfile(input_dir, '*.jpeg'))];

% 循环处理每个文件
for i = 1:length(file_list)
    filename = file_list(i).name;
    full_input_path = fullfile(input_dir, filename);
    
    % 读取图像
    img = im2double(imread(full_input_path));
    
    gamma = 2.2;
    RGB = 1;  % 这个变量似乎未使用，可能可以移除
    
    if size(img, 3) == 1
        f = img;
        gray = img;
    else
        hsv = rgb2hsv(img);
        f = hsv(:, :, 3);
        gray = rgb2gray(img);
    end
    
    I_ref = max(img, [], 3);
    R = img(:, :, 1);
    G = img(:, :, 2);
    B = img(:, :, 3);
    
    [RI, RR, Rw, Rstopterr] = newPR(R, I_ref);
    [GI, GR, Gw, Gstopterr] = newPR(G, I_ref);
    [BI, BR, Bw, Bstopterr] = newPR(B, I_ref);
    
    RI_gamma = RI.^(1/gamma);
    RS_gamma = RR .* RI_gamma;
    GI_gamma = GI.^(1/gamma);
    GS_gamma = GR .* GI_gamma;
    BI_gamma = BI.^(1/gamma);
    BS_gamma = BR .* BI_gamma;
    
    enhance(:, :, 1) = RS_gamma;
    enhance(:, :, 2) = GS_gamma;
    enhance(:, :, 3) = BS_gamma;
    
    % 保存增强图像到输出文件夹，使用相同的文件名
    full_output_path = fullfile(output_dir, filename);
    imwrite(enhance, full_output_path);
    
    % 可选：显示进度
    disp(['Processed: ' filename]);
end

disp('All images processed and saved.');