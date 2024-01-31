clear MatrixA

NumCols = input('Enter matrix dimension: ');

fprintf("Matrix dimension: %d\n", NumCols)

fprintf("Allocating matrix row-wise\n")
tic;
for i = 1:NumCols
    for j = 1:NumCols
    MatrixA(i,j) = 10*i+j;
    end
end
cpu_time_used_row = toc;
fprintf("Time for allocate row-wise: %f seconds\n", cpu_time_used_row);

clear MatrixA
fprintf("Allocating matrix column-wise\n")

tic;
for j = 1:NumCols
    for i = 1:NumCols
    MatrixA(i,j) = 10*i+j;
    end
end
cpu_time_used_col = toc;
fprintf("Time for allocate column-wise: %f seconds\n", cpu_time_used_col);
