__precompile__()
using BenchmarkTools, LinearAlgebra

function AddMatRowWise(MatrixA::Matrix{Float64})
    NumRows,NumCols = size(MatrixA)
    for  i = 1:NumRows
        for j = 1:NumCols
            MatrixA[i,j] = 10*i + j
        end
    end
end


function AddMatColWise(MatrixA::Matrix{Float64})
    NumRows,NumCols = size(MatrixA)
    for  j = 1:NumCols
        for i = 1:NumRows
            MatrixA[i,j] = 10*i + j
        end
    end
end

function askdimensions()
    println("Enter the number of rows:")
    NumRows = parse(Int64, readline())
    println("Enter the number of columns:")
    NumCols = parse(Int64, readline())
    return NumRows, NumCols
end

NumRows, NumCols = askdimensions()

MatrixA = Matrix{Float64}(undef,NumRows,NumCols)

cpu_time_used_row = @belapsed AddMatRowWise($MatrixA)

cpu_time_used_col = @belapsed AddMatColWise($MatrixA)


println("Time for allocate row-major: $cpu_time_used_row seconds")

println("Time for allocate column-major: $cpu_time_used_col seconds")

