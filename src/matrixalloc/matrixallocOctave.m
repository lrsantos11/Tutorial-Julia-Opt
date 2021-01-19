clear MatrixA

NumRows = 5000
NumCols = 5000
tic;
for i = 1:NumRows
    for j = 1:NumCols
    MatrixA(i,j) = 10*i+j;
    end
end
cpu_time_used_row = toc;
fprintf("Time for allocate row-wise: %f seconds\n", cpu_time_used_row);

clear MatrixA
for j = 1:NumCols
    for i = 1:NumRows
    MatrixA(i,j) = 10*i+j;
    end
end
cpu_time_used_col = toc;
fprintf("Time for allocate column-wise: %f seconds\n", cpu_time_used_col);
