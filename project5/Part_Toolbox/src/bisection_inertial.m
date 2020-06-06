function [part1,part2] = bisection_inertial(A,xy,picture)
% INERTPART : Inertial partition of a graph.
%
% p = inertpart(A,xy) returns a list of the vertices on one side of a partition
%     obtained by bisection with a line or plane normal to a moment of inertia
%     of the vertices, considered as points in Euclidean space.
%     Input A is the adjacency matrix of the mesh (used only for the picture!);
%     each row of xy is the coordinates of a point in d-space.
%
% inertpart(A,xy,1) also draws a picture.
%
% See also PARTITION

% 
% disp(' ');
% disp(' HPC 2020 @ ETH Zurich:   ');
% disp(' Implement inertial bisection');
% disp(' ');


% Steps
% 1. Calculate the center of mass.
% 2. Construct the matrix M.
%  (Consult the pdf of the assignment for the creation of M) 
% 3. Calculate the smallest eigenvector of M.  
% 4. Find the line L on which the center of mass lies.
% 5. Partition the points around the line L.
%   (you may use the function partition.m)


% <<<< Dummy implementation to generate a partitioning
% n   = size(A,1);
% map = zeros(n,1;
% map(1:round((n/2)))     = 0; 
% map((round((n/2))+1):n) = 1;
% [part1,part2] = other(map);
% Dummy implementation to generate a partitioning >>>>

% find xbar, ybar
% create M
% find smallest EV of M
% partition using the function provided
% iterate over result
% return partition

% probably the most expensive way to calculate the required things
n   = size(A,1);
x = xy(:,1);
y = xy(:,2);
xbar = sum(x) / n; % could use the mean function but for expressiveness
ybar = sum(y) / n;
x2 = sum(dot(x - ones(n,1)*xbar, x - ones(n,1)*xbar));
y2 = sum(dot(y - ones(n,1)*ybar, y - ones(n,1)*ybar));
x1y1 = sum(dot(x - ones(n,1)*xbar, y - ones(n,1)*ybar));

M = [y2, x1y1; x1y1, x2];

[ev, lam] = eigs(M, 1, 'smallestreal');
ev = ev/norm(ev);



[part1, part2] = partition(xy - [xbar,ybar], ev);

% n   = size(A,1);
% map = zeros(n,1);
% map(1:round((n/2)))     = 0; 
% map((round((n/2))+1):n) = 1;
% [part1,part2] = other(map);
% 

if picture == 1
    gplotpart(A,xy,part1);
    title('Inertial bisection');
end


end