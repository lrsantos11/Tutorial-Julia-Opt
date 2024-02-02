__precompile__()
# using BenchmarkTools

using LinearAlgebra

function AddMatRowWise!(MatrixA::Matrix{Float64})
    NumRows,NumCols = size(MatrixA)
    @inbounds for  i = 1:NumRows
        @inbounds for j = 1:NumCols
            MatrixA[i,j] = 10*i + j
        end
    end
end


function AddMatColWise!(MatrixA::Matrix{Float64})
    NumRows,NumCols = size(MatrixA)
    @inbounds for  j = 1:NumCols
        @inbounds for i = 1:NumRows
            MatrixA[i,j] = 10*i + j
        end
    end
end

function askdimensions()
	print("Enter matrix dimension: ");
    Num = parse(Int,readline())
    return Num
end

NumCols = askdimensions()

MatrixA = Matrix{Float64}(undef, NumCols, NumCols)

println("Matrix size: ", size(MatrixA))
println("Computing time for row-wise and column-wise allocation...")
cpu_time_used_row = @elapsed AddMatRowWise!(MatrixA)
cpu_time_used_col = @elapsed AddMatColWise!(MatrixA)


println("Time for allocate row-wise: $cpu_time_used_row seconds")

println("Time for allocate column-wise: $cpu_time_used_col seconds")

