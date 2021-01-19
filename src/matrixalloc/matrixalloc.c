#include <stdio.h>
#include <stdlib.h> 
#include <time.h>

void AddMatRowWise(double **MatrixA,int NumRows,int NumCols);
void AddMatColWise(double **MatrixA,int NumRows,int NumCols);

int main()
{
	int NumRows, NumCols;
    clock_t start, end;
    double cpu_time_used_row, cpu_time_used_col;


	printf("Enter rows and column for matrix: ");
	scanf("%d %d", &NumRows, &NumCols);
    
    printf("Number of rows: %d\nNumber of columns: %d\n", NumRows, NumCols);

    double **MatrixA = (double **)malloc(NumRows * sizeof(double *)); 
        for (int i=0; i<NumRows; i++) 
             MatrixA[i] = (double *)malloc(NumCols * sizeof(double)); 

    start = clock();
    AddMatRowWise(MatrixA,NumRows,NumCols);
    end = clock();
    cpu_time_used_row = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Time for allocate row-wise: %f seconds\n", cpu_time_used_row);

    start = clock();
    AddMatColWise(MatrixA,NumRows,NumCols);
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
