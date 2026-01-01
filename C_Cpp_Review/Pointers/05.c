#include <stdio.h>

int main() {
    int arr[] = {1,2,3,4,5};
    int * ptr = arr;
    printf("Position one: %d\n", *ptr);

    for(int i = 0; i < 5; i++) {
        printf("%d\t", *ptr);
        printf("%p\t", ptr);
        printf("%p\t\n", &ptr);
        ptr++;
    }
}