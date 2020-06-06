function Bench_bisection()
% Compare various graph bisection algorithms
%
% D.P & O.S for HPC2020 in ETH

% add the necessary paths
addpaths_GP;

%warning('off','all');
picture = 1;

format compact;

disp('          *********************************************')
disp('          ***      Graph bisection benchmark        ***');
disp('          *********************************************')
disp(' ');
disp(' The file "Toy_meshes.mat" contains sample meshes with coordinates.');
disp(' ');

% load meshes
load Toy_meshes;
whos;
names = ["grid5rect_10_100_", "grid5rect_100_10_", "gridt_40_",...
    "gridt9_30_", "Smallmesh", "Tapir", "Eppstein"];

for nmesh = 1:7
    close all; clf reset;
    
    if (nmesh==1)
        disp(' ');
        disp(' Function "grid5rec" produces a rectangular grid:');
        disp(' ');
        disp('[A,xy] = grid5rec(10,100);');
        disp(' ');
        [W,coords] = grid5rec(10, 100);
    end
    if (nmesh==2)
        disp(' ');
        disp(' Function "grid5rec" produces a rectangular grid:');
        disp(' ');
        disp('[A,xy] = grid5rec(100,10);');
        disp(' ');
        [W,coords] = grid5rec(100, 10);
    end   
    if (nmesh==3)
        disp(' ');
        disp(' Function "gridt" produces a triangular grid:');
        disp(' ');
        disp(' (See also grid5, grid7, grid9, grid3d, grid3dt.)');
        disp(' ');
        disp('[A,xy] = gridt(40);');
        disp(' ');
        [W,coords] = gridt(40);
    end
    if (nmesh==4)
        disp(' ');
        disp(' Function "gridt" produces a triangular grid:');
        disp(' ');
        disp(' (See also grid5, grid7, grid9, grid3d, grid3dt.)');
        disp(' ');
        disp('[A,xy] = grid9(30);');
        disp(' ');
        [W,coords] = grid9(30);
    end
    if (nmesh==5)
        W       = Smallmesh;
        coords = Smallmesh_coords;
    end
    if (nmesh==6)
        W       = Tapir;
        coords  = Tapir_coords;
    end

    if (nmesh==7)
        W       = Eppstein;
        coords  = Eppstein_coords;
    end
    
    
    disp(' ');
    disp('          *********************************************')
    disp('          ***        Various Bisection Methods      *** ');
    disp('          *********************************************')
    disp(' ');
    disp(' ');
    
    
    if (nmesh==1)
        disp('An initial rectangular  grid5rec(8,80) mesh');
    end
    if (nmesh==2)
        disp('An initial rectangular  grid5rec(80,8) mesh');
    end
    if (nmesh==3)
        disp('  gridt(20) mesh');
    end
    if (nmesh==4)
        disp('  gridt9(20) mesh');
    end
    if (nmesh==5)
        disp(' Small mesh ');
    end
    if (nmesh==6)
        disp(' "Tapir" is a test of a no-obtuse-angles mesh generation algorithm');
        disp(' due to Bern, Mitchell, and Ruppert.  ');
    end
    if (nmesh==7)
        disp('  Eppstein mesh');
    end
    
    figure(1)
    disp('gplotg(Tmesh,Tmeshxy);');
    disp('  ');
    gplotg(W,coords);
    nvtx  = size(W,1);
    nedge = (nnz(W)-nvtx)/2; 
    xlabel([int2str(nvtx) ' vertices, ' int2str(nedge) ' edges'],'visible','on');
    
    disp(' Hit space to continue ...');
    pause;
    
    disp(' 1. Coordinate bisection of a mesh.  ');
    disp(' p = coordpart(A,xy) returns a list of the vertices  ');
    disp(' on one side of a partition obtained by bisection    ');
    disp(' perpendicular to a coordinate axis.  We try every   ');
    disp(' coordinate axis and return the best cut. Input W is ');
    disp(' the adjacency matrix of the mesh; each row of xy is ');
    disp(' the coordinates of a point in d-space.              ');
    
    figure(2)
    [p1,p2] = bisection_coordinate(W,coords,picture);
    [cut_coord] = cutsize(W,p1);
    disp('Space to continue ...');
    %saveas(gcf, strcat('./bisection_coordinate_', names(nmesh), '.png'));
    pause;
    
    figure(3)
    disp(' ');
    disp(' 2. A multilevel method from the "Metis 5.0.2" package.');
    disp(' This will only work if you have Metis and its Matlab interface.');
    disp('  ');
    [p1,p2] = bisection_metis(W,coords,picture);
    [cut_metis] = cutsize(W,p1);
    disp('  ');
    disp(' Hit space to continue ...');
    %saveas(gcf, strcat('./bisection_metis_', names(nmesh), '.png'));
    pause;
        
            
    disp(' ');
    disp(' 3. Spectral partitioning, which uses the second eigenvector of');
    disp(' the Laplacian matrix of the graph, also known as the "Fiedler vector".');
    disp('  ');
    figure(6)
    [p1,p2] = bisection_spectral(W,coords,picture);    
    [cut_spectral] = cutsize(W,p1);
    disp('  ');    
    disp(' Hit space to continue ...');
    %saveas(gcf, strcat('./bisection_spectral_', names(nmesh), '.png'));
    pause;
    
    figure(7); 
    disp(' ');
    disp(' 4. Inertial partitioning, which uses the coordinates to find');
    disp(' a separating line in the plane.');
    disp('  ');
    [p1,p2] = bisection_inertial(W,coords,picture);    
    [cut_inertial] = cutsize(W,p1);
    disp('  ');
    disp(' Hit space to continue ...');
    pause;
    %saveas(gcf, strcat('./bisection_inertial_', names(nmesh), '.png'));
    close all;
    
    format;
    
    disp(' ');
    disp('          ************************************************')
    disp('          ***        Bisection Benchmark               *** ');
    disp('          ***         D.P. & O.S. for HPC, ETH Zurich   *** ');
    disp('          ************************************************')
    disp(' ');
    disp(' ');
    
    
end