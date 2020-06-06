function [part1,part2] = bisection_spectral(A,xy,picture)
% bisection_spectral : Spectral partition of a graph.
%
% [part1,part2] = bisection_spectral(A) returns a partition of the n vertices
%                 of A into two lists part1 and part2 according to the
%                 spectral bisection algorithm of Simon et al:  
%                 Label the vertices with the components of the Fiedler vector
%                 (the second eigenvector of the Laplacian matrix) and partition
%                 them around the median value or 0.


% 
% disp(' ');
% disp(' HPC 2020 @ ETH Zurich:   ');
% disp(' Implement spectral bisection');
% disp(' ');


% Steps
% 1. Construct the Laplacian.
% 2. Calculate its eigensdecomposition.
% 3. Label the vertices with the components of the Fiedler vector.
% 4. Partition them around their median value, or 0.




% <<<< Dummy implementation to generate a partitioning
% n = size(A,1);
% map = zeros(n,1);
% map(1:round((n/2))) = 0; 
% map((round((n/2))+1):n) = 1;
% [part1,part2] = other(map);
% 
% if picture == 1
%     gplotpart(A,xy,part1);
%     title('Spectral bisection (dummy) using the Fiedler Eigenvector');
% end

% Dummy implementation to generate a partitioning >>>>
 
% <<<< real implemenation
n = size(A,1);
map = zeros(n,1);

Adj = graph(A);
% remove all self-loops to be able to continuously use laplacian()

% alternatively we could form L by hand: 
% D = diag(sum(A,2)); L=D-A;
Adj = rmedge(Adj, 1:n, 1:n);
Lapl = laplacian(Adj);
[ev, lam] = eigs(Lapl, 2, 'smallestreal');
fiedler = ev(:,2);
m = median(fiedler);
for l = 1:n
    if fiedler(l) < m
        map(l) = 1;
    end
end
[part1, part2] = other(map);

if picture == 1
    gplotpart(A,xy,part1);
    title('Spectral bisection using the Fiedler Eigenvector');
end

end