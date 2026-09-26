#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define N 9

void print_grid(int arr[N][N])
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%d ", arr[i][j]);
        printf("\n");
    }
}

int isSafe(int grid[N][N], int row, int col, int num)
{
    for (int x = 0; x < N; x++)
        if (grid[row][x] == num)
            return 0;

    for (int x = 0; x < N; x++)
        if (grid[x][col] == num)
            return 0;

    int startRow = row - row % 3;
    int startCol = col - col % 3;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (grid[i + startRow][j + startCol] == num)
                return 0;

    return 1;
}

/*
 * Versão sequencial usada pelas tasks depois do primeiro nível.
 * Cada task recebe sua própria cópia da grade, eliminando
 * condições de corrida.
 */
int solveSudoku_seq(int grid[N][N], int row, int col)
{
    if (row == N - 1 && col == N)
        return 1;

    if (col == N) {
        row++;
        col = 0;
    }

    if (grid[row][col] > 0)
        return solveSudoku_seq(grid, row, col + 1);

    for (int num = 1; num <= N; num++) {
        if (isSafe(grid, row, col, num)) {
            grid[row][col] = num;
            if (solveSudoku_seq(grid, row, col + 1))
                return 1;
            grid[row][col] = 0;
        }
    }
    return 0;
}

static volatile int solved = 0;
static int solution[N][N];

void solveSudoku_parallel(int grid[N][N], int row, int col)
{
    if (solved) return;

    if (row == N - 1 && col == N) {
#pragma omp critical
        {
            if (!solved) {
                solved = 1;
                memcpy(solution, grid, sizeof(solution));
            }
        }
        return;
    }

    if (col == N) {
        row++;
        col = 0;
    }

    if (grid[row][col] > 0) {
        solveSudoku_parallel(grid, row, col + 1);
        return;
    }

    for (int num = 1; num <= N; num++) {
        if (solved) return;

        if (isSafe(grid, row, col, num)) {
            /* Copia a grade para a task — sem estado compartilhado */
            int local_grid[N][N];
            memcpy(local_grid, grid, sizeof(local_grid));
            local_grid[row][col] = num;

#pragma omp task firstprivate(local_grid, row, col) shared(solved, solution)
            {
                if (!solved)
                    /* continua sequencialmente dentro da task */
                    if (solveSudoku_seq(local_grid, row, col + 1)) {
#pragma omp critical
                        {
                            if (!solved) {
                                solved = 1;
                                memcpy(solution, local_grid,
                                       sizeof(solution));
                            }
                        }
                    }
            }
        }
    }

#pragma omp taskwait
}

int main(void)
{
    int grid[N][N] = {
        {5, 3, 0, 0, 7, 0, 0, 0, 0},
        {6, 0, 0, 1, 9, 5, 0, 0, 0},
        {0, 9, 8, 0, 0, 0, 0, 6, 0},
        {8, 0, 0, 0, 6, 0, 0, 0, 3},
        {4, 0, 0, 8, 0, 3, 0, 0, 1},
        {7, 0, 0, 0, 2, 0, 0, 0, 6},
        {0, 6, 0, 0, 0, 0, 2, 8, 0},
        {0, 0, 0, 4, 1, 9, 0, 0, 5},
        {0, 0, 0, 0, 8, 0, 0, 7, 9}
    };

#pragma omp parallel
    {
#pragma omp single
        {
            solveSudoku_parallel(grid, 0, 0);
        }
    }

    if (solved) {
        printf("Solucao encontrada:\n");
        print_grid(solution);
    } else {
        printf("Nenhuma solucao existe.\n");
    }

    return 0;
}