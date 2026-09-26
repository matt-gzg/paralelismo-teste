#include <stdio.h>
#include <omp.h>

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int arr[], int low, int high) {
    int p = arr[low];
    int i = low;
    int j = high;

    while (i < j) {
        while (arr[i] <= p && i <= high - 1) {
            i++;
        }
        while (arr[j] > p && j >= low + 1) {
            j--;
        }
        if (i < j) {
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[low], &arr[j]);
    return j;
}

#define MAX_DEPTH 4

void quickSort(int arr[], int low, int high, int depth) {
    if (low < high) {
        int pi = partition(arr, low, high);

        if (depth < MAX_DEPTH) {
            #pragma omp task shared(arr) firstprivate(low, pi, depth)
            quickSort(arr, low, pi - 1, depth + 1);

            #pragma omp task shared(arr) firstprivate(pi, high, depth)
            quickSort(arr, pi + 1, high, depth + 1);

            #pragma omp taskwait
        } else {
            quickSort(arr, low, pi - 1, depth + 1);
            quickSort(arr, pi + 1, high, depth + 1);
        }
    }
}

int main() {
    int arr[] = { 4, 2, 5, 3, 1 };
    int n = sizeof(arr) / sizeof(arr[0]);

    #pragma omp parallel
    {
        #pragma omp single
        {
            quickSort(arr, 0, n - 1, 0);
        }
    }

    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");

    return 0;
}