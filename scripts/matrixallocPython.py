# matrix allocation row-wise and column-wise



import numpy as np
from timeit import default_timer as timer


def AddMatRowWise(MatrixA):
    NumRows, NumCols = MatrixA.shape
    for i in range(NumRows):
        for j in range(NumCols):
            MatrixA[i,j] = 10*i + j

def AddMatColWise(MatrixA):
    NumRows, NumCols = MatrixA.shape
    for j in range(NumCols):
        for i in range(NumRows):
            MatrixA[i,j] = 10*i + j

def askdimensions():
    NumRows = int(input("Enter number of rows: "))
    NumCols = int(input("Enter number of columns: "))
    return NumRows, NumCols

NumRows, NumCols = askdimensions()

MatrixA = np.empty((NumRows, NumCols))

repeats = 10


# 
cpu_time_used_row = 0.0


for i in range(repeats):
    start = timer()
    AddMatRowWise(MatrixA)
    end = timer()
    cpu_time_used_row = cpu_time_used_row + (end - start)

cpu_time_used_row = cpu_time_used_row / repeats

cpu_time_used_col = 0.0
for i in range(repeats):
    start = timer()
    AddMatColWise(MatrixA)
    end = timer()
    cpu_time_used_col = cpu_time_used_col + (end - start)
cpu_time_used_col = cpu_time_used_col / repeats


print(f"Time for allocate row-major: {cpu_time_used_row:.6f} seconds")
print(f"Time for allocate column-major: {cpu_time_used_col:.6f} seconds")
