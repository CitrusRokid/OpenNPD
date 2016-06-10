function depth = CalcTreeDepth(leftChild, rightChild, node)
% function to calculate the depth of the learned tree.

if nargin < 3
    node = 0;
end

if node + 1 > length(leftChild)
    depth = 0;
    return;
end

if leftChild(node + 1) < 0
    ld = 0;
else
    ld = CalcTreeDepth(leftChild, rightChild, leftChild(node + 1));
end

if rightChild(node + 1) < 0
    rd = 0;
else
    rd = CalcTreeDepth(leftChild, rightChild, rightChild(node + 1));
end

depth = max(ld, rd) + 1;
end
