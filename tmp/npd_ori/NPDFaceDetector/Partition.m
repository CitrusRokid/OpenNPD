function [label, nGroups] = Partition(A)
%% function label = Partition(A)
% Partition of N elements in a set according to the adjacency matrix A,
% return labels of each element, and the number of groups.
% 
% example: 
% A = [1 0 1 0 0 0 0 0 0; 0 1 0 1 0 0 0 0 0; 1 0 1 0 0 0 0 0 0; 0 1 0 1 1 0 0 0 1; 
%      0 0 0 1 1 0 1 1 0; 0 0 0 0 0 1 0 0 0; 0 0 0 0 1 0 1 0 0; 0 0 0 0 1 0 0 1 0; 0 0 0 1 0 0 0 0 1]
% [label, nGroups] = Partition(A)
%
% scliao@nlpr.ia.ac.cn
%

%% make set
N = size(A,1);
parent = 1 : N;
rank = zeros(N,1);

for i = 1 : N
    %% check equal items
    for j = 1 : N
        if A(i,j) == 0
            continue;
        end
        
        %% find root of node i and compress path
        [root_i, parent] = Find(parent, i);    
        
        %% find root of node j and compress path
        [root_j, parent] = Find(parent, j);
        
        %% union both trees 
        if root_j ~= root_i
            if rank(root_j) < rank(root_i)
                parent(root_j) = root_i;
            else if rank(root_i) < rank(root_j)
                    parent(root_i) = root_j;
                else
                    parent(root_j) = root_i;
                    rank(root_i) = rank(root_i) + 1;
                end
            end
        end
    end
end

%% label each element
flag = parent == 1:N;
nGroups = sum(flag);
label = zeros(N,1);
label(flag) = 1 : nGroups;

for i = 1 : N
    if parent(i) == i
        continue;
    end
    
    % find root of node i
    root_i = Find(parent, i);    
    label(i) = label(root_i);
end


function [root, parent] = Find(parent, x)
%% find root of node x and compress path
root = parent(x);
if root ~= x
    [root, parent] = Find(parent, root);
end
