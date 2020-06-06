% Visualize information from the eigenspectrum of the graph Laplacian
%
% D.P & O.S for HPC2020 in ETH


% add necessary paths
addpaths_GP;

% Graphical output at bisection level
picture = 0;

% Cases under consideration
%load airfoil1.mat;

%load 3elt.mat;
%load barth4.mat;


load mesh3e1.mat;
%load crack.mat;

% Initialize the cases
W      = Problem.A;
coords = Problem.aux.coord;

% Steps
% 1. Construct the graph Laplacian of the graph in question.
adj = graph(W);
% again need to purge self loops here else the laplacian is ill-defined
adj = rmedge(adj, 1:length(W), 1:length(W));
lapl = laplacian(adj);
% 2. Compute eigenvectors associated with the smallest eigenvalues.
format long e;

[ev, lam] = eigs(lapl, 3, 'smallestreal');
disp(lam);
second_ev = ev(:,2); second_lam = lam(2,2);
% 3. Perform spectral bisection.

[p1, p2] = bisection_spectral(W, coords, 0);
% 4. Visualize:
%   i.   The first and second eigenvectors.
%plot(1:size(W, 1), ev(:,1), 1:size(W, 1), ev(:,2), 1:size(W, 1), ones(size(W,1))*median(ev(:,2))), legend('smallest EV', 'Fiedler','Fiedler median');


% disp(lam);
%figure(1);
%colordef(gcf,'black');

figure(1);
plot(1:size(W, 1), ev(:,1), 1:size(W, 1), ev(:,2)), legend('smallest EV', 'Fiedler');
% 
% 
% pause;

% %   ii.  The second eigenvector projected on the coordinate system space of graphs.

N = size(W,1);

zbounds = [min(ev(:,2)), max(ev(:,2))];
figure(2);
gplotpart_proj(W,[coords,zeros(length(coords),1)],zbounds,p1,[.3,.3,.3],'black');
hold on;
scatter3(coords(:,1),coords(:,2),ev(:,2),80*ones(N,1),ev(:,2), '.');
colorbar;
axis on;
%set(gca,'BoxStyle','full','Box','on');
%set(gca, 'PlotBoxAspectRatio', [1 1 1]);
set(gca, 'DataAspectRatio', [20 20 .4], 'PlotBoxAspectRatio', [2 1.5 2.5]);
view(-10, 10);
%hold off;

pause;

%figure(2);


% hold on;
% a = gcf;
% disp(a);
% xd = get(a, 'XData');
% yd = get(a, 'YData');
% zd = zeros(length(xd));
% 
% s = ones(size(W, 1), 1);
% color = ev(:,2);
% s1 = scatter3(coords(:,1),coords(:,2),ev(:,2),s,color);
% figure(2);
% hold on;

%   iii. The spectral bi-partitioning results using the spectral
%   coordinates of each graph.

% clf reset
% colordef(gcf,'black')
% set(gcf, 'InvertHardcopy', 'off')
% hold on
figure(3);

%functions to generate the last graphics


%gplotpart(W, [-ev(:,2), ev(:,3)], p1, [.4 .4 .4], 'white');
gplotpart(W, coords, p1, [.4 .4 .4], 'white');
set(gcf, 'InvertHardcopy', 'off')
saveas(gcf, strcat('./normal_mesh3e1', '.png'));
hold on;
pause;






