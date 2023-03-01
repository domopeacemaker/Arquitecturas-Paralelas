#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <unistd.h>

#define SIZE 16
#define UNASSIGNED 0
double start;

void print_grid(int grid[SIZE][SIZE]) {
    for (int row = 0; row < SIZE; row++) {
        for (int col = 0; col < SIZE; col++) {
            printf("%4d", grid[row][col]);
        }
        printf("\n");
    }
}

int is_exist_row(int grid[SIZE][SIZE], int row, int num){
    for (int col = 0; col < 16; col++) {
        if (grid[row][col] == num) {
            return 1;
        }
    }
    return 0;
}

int is_exist_col(int grid[SIZE][SIZE], int col, int num) {
    for (int row = 0; row < 16; row++) {
        if (grid[row][col] == num) {
            return 1;
        }
    }
    return 0;
}

int is_exist_box(int grid[SIZE][SIZE], int startRow, int startCol, int num) {
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            if (grid[row + startRow][col + startCol] == num) {
                return 1;
            } 
        }
    }
    return 0;
}

int is_safe_num(int grid[SIZE][SIZE], int row, int col, int num) {
    return !is_exist_row(grid, row, num) 
            && !is_exist_col(grid, col, num) 
            && !is_exist_box(grid, row - (row % 4), col - (col %4), num);
}

int find_unassigned(int grid[SIZE][SIZE], int *row, int *col) {
    for (*row = 0; *row < SIZE; (*row)++) {
        for (*col = 0; *col < SIZE; (*col)++) {
            if (grid[*row][*col] == UNASSIGNED) {
                return 1;
            }
        }
    }
    return 0;
}

int solve(int grid[SIZE][SIZE], int level) {
    
    int row = 0;
    int col = 0;
    if (!find_unassigned(grid, &row, &col)) return 1; 
    
    for (int num = 1; num <= SIZE; num++ ) {        
        if (is_safe_num(grid, row, col, num)) {            		
            #pragma omp task default(none) firstprivate(grid, row, col, num, level) shared(start) final(level>1)
            {			
		int copy_grid[SIZE][SIZE];			
		memcpy(copy_grid,grid,SIZE*SIZE*sizeof(int));				                
		copy_grid[row][col] = num;          
                if(solve(copy_grid, level+1)) {
                    double end = omp_get_wtime();
                    double time_spent = end - start;  
                    print_grid(copy_grid);
                    printf("\nTime: %f s\n",time_spent);  
                    exit(0);                    
                }
            }                       
                   
        }
    }
    #pragma omp taskwait
    return 0;
}

int main(int argc, char** argv) 
{
    int sudoku[SIZE][SIZE]={
        {0, 6, 0, 0, 0, 0, 0, 8, 11, 0, 0, 15, 14, 0, 0, 16},
        {15, 0, 0, 0, 0, 16, 14, 0, 0, 0, 12, 0, 0, 6, 0, 0},
 	{13, 0, 9, 12, 0, 0, 0, 0, 0, 16, 14, 0, 15, 11, 10, 0},
 	{2, 0, 16, 0, 11, 0, 15, 10, 1, 0, 0, 0, 0, 0, 0, 0},
 	{0, 15, 11, 10, 0, 0, 16, 2, 13, 8, 9, 12, 0, 0, 0, 0},
 	{12, 13, 0, 0, 4, 1, 0, 6, 2, 3, 0, 0, 0, 0, 11, 10},
 	{5, 0, 6, 1, 12, 0, 9, 0, 15, 11, 0, 7, 16, 0, 0, 3},
 	{0, 2, 0, 0, 0, 10, 0, 11, 6, 0, 5, 0, 0, 13, 0, 9},
 	{10, 7, 15, 11, 16, 0, 0, 0, 12, 13, 0, 0, 0, 0, 0, 6},
        {9, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 16, 10, 0, 0, 11},
        {1, 0, 4, 6, 9, 13, 0, 0, 7, 0, 11, 0, 0, 16, 0, 0},
        {16, 14, 0, 0, 7, 0, 10, 15, 0, 6, 1, 0, 0, 0, 13, 8},
        {11, 10, 0, 15, 0, 0, 0, 16, 0, 12, 13, 0, 0, 1, 5, 4},
        {0, 0, 12, 0, 0, 4, 6, 0, 16, 0, 0, 0, 11, 10, 0, 0},
        {0, 0, 5, 0, 8, 12, 13, 0, 10, 0, 0, 11, 2, 0, 0, 14},
        {3, 16, 0, 0, 10, 0, 0, 7, 0, 0, 6, 0, 0, 0, 12, 0} };

    printf("Size: %d", SIZE);   
    printf("\n");
        
    printf("Solving Sudoku: \n");
    print_grid(sudoku);
    printf("---------------------\n");   	   

    start = omp_get_wtime();
    #pragma omp parallel default(none) shared(sudoku)
    #pragma omp single nowait
    {
       solve(sudoku,1);   
    }
}
