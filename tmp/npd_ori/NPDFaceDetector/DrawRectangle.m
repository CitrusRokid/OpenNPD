function img = DrawRectangle(img, col, row, width, height, color, border)
if nargin < 7
    border = 1;
    if nargin < 6
        color = [0 255 0];
    end
end

[h,w,c] = size(img);

% check border
if col < 1 || col + width - 1 > w || row < 1 || row + height - 1 > h
    warning('MATLAB:ErrorSize', 'Error size of rectangle.');
    col = max(col,1);
    row = max(row,1);
    width = min(width, w - col + 1);
    height = min(height, h - row + 1);
end

if c == 1
    img = repmat(img, [1,1,3]);
end

for i = 1 : 3
    img(row : row + border - 1, col : col + width - 1, i) = color(i);
    img(row + height - border : row + height - 1, col : col + width - 1, i) = color(i);
    img(row : row + height - 1, col : col + border - 1, i) = color(i);    
    img(row : row + height - 1, col + width - border : col + width - 1, i) = color(i);
end
