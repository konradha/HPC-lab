function [cut_recursive,cut_kway] = Bench_metis(picture)
% Compare recursive bisection and direct k-way partitioning,
% as implemented in the Metis 5.0.2 library.

%  Add necessary paths
addpaths_GP;

% Graphs in question
load usroads;
load luxembourg_osm;

fprintf('       *********************************************\n')
fprintf('       ***  Metis bisection benchmark  ***\n');
fprintf('       *********************************************\n')

cases = {
    'usroads.mat',
    'luxembourg_osm.mat'
    };

names = {'usroads', 'luxembourg_osm'};
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
fprintf(repmat('-', 1, 45));
fprintf('\n');
for c = 1:nc
    spacers  = repmat('.', 1, maxlen+3-length(cases{c}));
    [params] = Initialize_case(sparse_matrices(c));
    fprintf('%s %s %10d %10d\n', cases{c}, spacers,params.numberOfVertices,params.numberOfEdges);
end

for c = 1:nc
    spacers = repmat('.', 1, maxlen+3-length(cases{c}));
    fprintf('%s %s', cases{c}, spacers);
    sparse_matrix = load(cases{c});
    

    % Recursively bisect the loaded graphs in 8 and 16 subgraphs.
    % Steps
    % 1. Initialize the problem
    ps = [8,16,32];
    
    for n=1:length(ps)
        [params] = Initialize_case(sparse_matrices(c));
        W      = params.Adj;
        coords = params.coords;

        [map, edgecut] = metismex('PartGraphRecursive',W,ps(n));
        figure(1);
        gplotmap(W,coords,map);
        saveas(gcf, strcat('./', int2str((ps(n))), '_recursive_', names{c}, '.png'));
        pause;
        
        

        [map, edgecut] = metismex('PartGraphKWay',W,ps(n));
        figure(2);
        gplotmap(W,coords,map);
        saveas(gcf, strcat('./', int2str((ps(n))), '_kway_', names{c}, '.png'));
        pause;
    end
    
    
    %[map, edgecut] = metismex('PartGraphKWay',W,p);
end

% Steps
% 1. Initialize the cases
% 2. Call metismex to
%     a) Recursively partition the graphs in 16 and 32 subsets.
%     b) Perform direct k-way partitioning of the graphs in 16 and 32 subsets.
% 3. Visualize the results for 32 partitions


end