% Benchmark for recursively partitioning meshes, based on various
% bisection approaches
%
% D.P & O.S for HPC2020 in ETH



% add necessary paths
addpaths_GP;
nlevels_a = 3;
nlevels_b = 4;

fprintf('       *********************************************\n')
fprintf('       ***  Recursive graph bisection benchmark  ***\n');
fprintf('       *********************************************\n')

% load cases
cases = {
    'mesh3e1.mat';
    'airfoil1.mat';
    '3elt.mat';
    'barth4.mat';
    'crack.mat';
    };

names = {
    'mesh3e1';
    'airfoil1';
    '3elt';
    'barth4';
    'crack';
    };

nc = length(cases);
maxlen = 0;
for c = 1:nc
    if length(cases{c}) > maxlen
        maxlen = length(cases{c});
    end
end

for c = 1:nc
    fprintf('.');
    sparse_matrices(c) = load(cases{c});
end


fprintf('\n\n Report Cases         Nodes     Edges\n');
fprintf(repmat('-', 1, 40));
fprintf('\n');
for c = 1:nc
    spacers  = repmat('.', 1, maxlen+3-length(cases{c}));
    [params] = Initialize_case(sparse_matrices(c));
    fprintf('%s %s %10d %10d\n', cases{c}, spacers,params.numberOfVertices,params.numberOfEdges);
end

%% Create results table
fprintf('\n%7s %16s %20s %16s %16s\n','Bisection','Spectral','Metis 5.0.2','Coordinate','Inertial');
fprintf('%10s %10d %6d %10d %6d %10d %6d %10d %6d\n','Partitions',8,16,8,16,8,16,8,16);
fprintf(repmat('-', 1, 100));
fprintf('\n');

set(gca,'Color','none');

for c = 1:nc
    spacers = repmat('.', 1, maxlen+3-length(cases{c}));
    fprintf('%s %s', cases{c}, spacers);
    sparse_matrix = load(cases{c});
    

    % Recursively bisect the loaded graphs in 8 and 16 subgraphs.
    % Steps
    % 1. Initialize the problem

    p = 3;
    [params] = Initialize_case(sparse_matrices(c));
    W      = params.Adj;
    coords = params.coords;
    
    %8/16 partitioned-graphs
    
    data = ones(8, 1);
    count = 1;
    names = ["spectral", "metis", "coordinate", "inertial"];
    meths = {@bisection_spectral, @bisection_metis, @bisection_coordinate,@bisection_inertial};
    for o = 1:length(meths)
        for p=3:4
            [m, s, A] = rec_bisection(meths{o}, p, W, coords, 0);
            [i,j] = find(W);
            f = find(m(i) > m(j));
            [a] = length(f);
            data(count) = a;
            count = count + 1;
        end
    end
    fprintf('%6d %6d %10d %6d %10d %6d %10d %6d\n',[data(:)]);


    
%     [map, cut] =rec_bisection(@bisection_metis, p, W, coords, 0);
%     figure(1);
%     gplotmap(W, coords, map);
%     set(gca,'Color','black');
%     %saveas(gcf, strcat('./', int2str((2^p)), 'bisection_metis_', names{c}, '.png'));
%     %saveas(gcf, strcat('./bisection_spectral_', names(nmesh), '.png'));
%     pause;
    
    
    % 2. Recursive routines
    % i. Spectral   
%     [m1, s1, A1] = rec_bisection(@bisection_spectral, p, W, coords, 0);
%     figure(1);
%     gplotmap(W,coords,m1);
%     title('Spectral recursive bisection');
%     % Count the separating edges
%     [i,j] = find(W);
%     f = find(m1(i) > m1(j));
%     [a1] = int2str(length(f));
%     saveas(gcf, strcat('./', int2str((2^p)), '_spectral', names{c}, '.png'));
%     pause;
%     % ii. Metis
%     [m2, s2, A2] = rec_bisection(@bisection_metis, p, W, coords, 0);
%     figure(2);
%     gplotmap(W,coords,m2 );
%     title('Metis recursive bisection');
%     saveas(gcf, strcat('./', int2str((2^p)), '_metis_', names{c}, '.png'));
%     pause;
%     % iii. Coordinate    
%     [m3, s3, A3] = rec_bisection(@bisection_coordinate, p, W, coords, 0);
%     figure(3);
%     gplotmap(W,coords,m3);
%     title('Coordinate recursive bisection');
%     saveas(gcf, strcat('./', int2str((2^p)), '_coordinate_', names{c}, '.png'));
%     pause;
%     % iv. Inertial
%     [m4, s4, A4] = rec_bisection(@bisection_inertial, p, W, coords, 0);
%     figure(4);
%     gplotmap(W,coords,m4);
%     title('Inertial recursive  bisection');
%     saveas(gcf, strcat('./', int2str((2^p)), '_inertial_', names{c}, '.png'));
%     pause;
%     % 3. Calculate number of cut edges
% 
%     % 4. Visualize the partitioning result
    
    
%     fprintf('%6d %6d %10d %6d %10d %6d %10d %6d\n',0,0,...
%     0,0,0,0,0,0);
    
end