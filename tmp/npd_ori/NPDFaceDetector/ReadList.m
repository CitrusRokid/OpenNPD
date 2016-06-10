function [numItems, filenames] = ReadList( listFile )

%{
function [numItems, filenames] = ReadList( listFile )

input :
    listFile    : list of image filenames

output :
    numItems  : number of total items contained in the list file
    filenames   : cell array of image filenames read from listFile

author:
    Shengcai Liao
    Email: scliao@cbsr.ia.ac.cn

date:
    03/03/2005
%}

fin = fopen( listFile, 'rt');
if fin < 0
    filenames = [];
    numItems = -1;
    return;
end

% A = textscan( fin, '%s', '\n' ); % in this way filenames containing white spaces cannot be correctly handled
% filenames = A{1};

filenames = {};
while ~feof(fin)
    filenames{end+1} = fgetl(fin); %#ok<AGROW>  % in this way filenames containing white spaces can be correctly handled
end

numItems = length(filenames);
fclose( fin );
