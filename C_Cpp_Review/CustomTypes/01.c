#include <stdio.h>
#include <stdlib.h>

int main() {
    int arr[] = {1,2,3,4,5};

    size_t size = sizeof(arr) / sizeof(arr[0]);
    printf("Size of arr:%zu\n", size);
    printf("Size of size:%zu\n", sizeof(size));
    printf("int size in bytes:%zu\n", sizeof(int));
}