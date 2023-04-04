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

    // Ask for dimensios
	printf("Enter the number of rows : ");
	scanf("%d", &NumRows);
    printf("Enter the number of columns : ");
    scanf("%d", &NumCols);
    
    // Allocate memory for the matrix

    double **MatrixA = (double **)malloc(NumRows * sizeof(double *)); 
        for (int i=0; i<NumRows; i++) 
             MatrixA[i] = (double *)malloc(NumCols * sizeof(double)); 
    
    int repeats = 10;

    // Repeats the operation to get a better time row-wise
    cpu_time_used_row = 0.0;

    for (int repeat = 0; repeat < repeats; repeat++)
    {
             start = clock();
             AddMatRowWise(MatrixA, NumRows, NumCols);
             end = clock();
             cpu_time_used_row += ((double)(end - start)) / CLOCKS_PER_SEC;
    }

    cpu_time_used_row = cpu_time_used_row / ((double) repeats);

    printf("Time for allocate row-major: %f seconds\n", cpu_time_used_row);


    // Repeats the operation to get a better time column-wise
    cpu_time_used_col = 0.0;
    for (int repeat = 0; repeat < repeats; repeat++)
    {
             start = clock();
             AddMatColWise(MatrixA, NumRows, NumCols);
             end = clock();
             cpu_time_used_col += ((double)(end - start)) / CLOCKS_PER_SEC;
    }

    cpu_time_used_col = cpu_time_used_col / ((double) repeats);

    printf("Time for allocate column-major: %f seconds\n", cpu_time_used_col);

    // Free memory    
    free(MatrixA);
	return 0;
}

void AddMatRowWise(double **MatrixA,int NumRows,int NumCols)
{
    for(int i = 0; i < NumRows; i++)
        for(int j = 0; j < NumCols; j++)
            MatrixA[i][j] = 10 * i + j;
            
}


void AddMatColWise(double **MatrixA,int NumRows,int NumCols)
{
        for(int j=0;j<NumCols;j++)
            for(int i=0;i<NumRows;i++)
                MatrixA[i][j]=10*i+j;
}
