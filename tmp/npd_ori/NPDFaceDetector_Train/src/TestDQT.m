function score = TestDQT(tree, x)
% function to test the learned DQT based weak classifier.

n = size(x,1);
score = zeros(n,1,'single');

if isempty(x)
    score(:) = repmat(tree.fit, size(x,1), 1);
else
   score = TestSubTree(tree, x, 0);
end
end


function score = TestSubTree(tree, x, node)

if isempty(x)
    score = [];
    return;
end

n = size(x,1);
score = zeros(n,1,'single');

if node < 0 % leaf node
    score(:) = tree.fit(-node);
else % branch node
    mNode = node + 1;
    
    isLeft = (x(:, mNode) < tree.cutpoint(1, mNode)) | (x(:, mNode) > tree.cutpoint(2, mNode));    
    score(isLeft) = TestSubTree(tree, x(isLeft,:), tree.leftChild(mNode));
    score(~isLeft) = TestSubTree(tree, x(~isLeft,:), tree.rightChild(mNode));
end
end
