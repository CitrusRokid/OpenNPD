function [Fx, passCount] = TestGAB(model, X)
% function to test the learned soft cascade and DQT weak classifier based Gentle AdaBoost classifier.

n = size(X, 1);
Fx = zeros(n, 1, 'single');
passCount = zeros(n, 1);

for t = 1 : length(model)
    index = passCount == t - 1;
    if isempty(index)
        break;
    end
    
    fx = TestDQT( model(t), X(index, model(t).feaId) );
    Fx(index) = Fx(index) + fx;
    passCount(index) = passCount(index) + double(Fx(index) >= model(t).threshold);
end
