function [h, display_array] = displayData(X, num_total, num_col)
%DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(X) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the displayed array if requested.

%num_row = ceil(num_total / num_col);
%[num_example height_example] = size(X);

% Gray Image
%colormap(gray);

% between images padding
%pad = 0;

display_array = X';
% Setup blank display
%display_array = -ones(num_row * (pad + height_example) - pad, ...
%		      num_col * (pad + 1) - pad);

% copy each example into display_array
%curr = 1;
%for i = 1:(pad+height_example):size(display_array,1)
%	for j = 1:(pad+1):size(display_array,2)
%		display_array(i:(i+height_example-1), j) = X(curr++, :)';
%	end
%end

% Display Image
h = imshow(display_array);

% Do not show axis
axis image off;

drawnow;

end
