#include <stdlib.h>
#include <stdio.h>

int main() {
    int * ptr = NULL;
    printf("1. Initial ptr value: %p\n", (void*)ptr);
    
    if (ptr ==NULL) {
        printf("ptr is NULL, cannot dereference\n");
    }

    ptr = malloc(sizeof(int));
    if (ptr == NULL) { 
        printf("Memory allocation failed\n");
        return 1;
    }
}