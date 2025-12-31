#include <stdio.h>

int main() {
    int value = 42;
    int * ptr = &value;
    int ** ptr2 = &ptr;

    printf("%p\n", ptr);
    printf("%p\n", ptr2);
    printf("%d\n", **ptr2);
}