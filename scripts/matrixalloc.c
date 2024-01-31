#include <stdio.h>
#include <stdlib.h> 
#include <time.h>

void AddMatRowWise(double **MatrixA,int NumRows,int NumCols);
void AddMatColWise(double **MatrixA,int NumRows,int NumCols);

int main()
{
	int NumCols;
    clock_t start, end;
    double cpu_time_used_row, cpu_time_used_col;


	printf("Enter matrix dimension: ");
	scanf("%d",  &NumCols);

    printf("Matrix Dimension  %d x %d\n", NumCols, NumCols);

    double **MatrixA = (double **)malloc(NumCols * sizeof(double *)); 
        for (int i=0; i<NumCols; i++) 
             MatrixA[i] = (double *)malloc(NumCols * sizeof(double)); 

    start = clock();
    AddMatRowWise(MatrixA,NumCols,NumCols);
    end = clock();
    cpu_time_used_row = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Time for allocate row-wise: %f seconds\n", cpu_time_used_row);

    start = clock();
    AddMatColWise(MatrixA,NumCols,NumCols);
    end = clock();
    cpu_time_used_row = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Time for allocate column-wise: %f seconds\n", cpu_time_used_row);

    free(MatrixA);
	return 0;
}

void AddMatRowWise(double **MatrixA,int NumRows,int NumCols)
{
        for(int i=0;i<NumRows;i++)
            for(int j=0;j<NumCols;j++)
                MatrixA[i][j]=10*i+j;
}

void AddMatColWise(double **MatrixA,int NumRows,int NumCols)
{
        for(int j=0;j<NumCols;j++)
            for(int i=0;i<NumRows;i++)
                MatrixA[i][j]=10*i+j;
}
