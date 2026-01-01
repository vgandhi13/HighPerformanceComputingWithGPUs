#include <stdio.h>

int main () {
    int num = 10;
    float fnum = 3.14;
    void *vptr;

    vptr = &num;
    printf("Integer: %d\n", *(int*)vptr); // output 10
    // vptr is a memory address "&num" but it is stored as void ptr . So we cast void pointer to integer type and then dereference that

    vptr = &fnum;
    printf("Float: %.2f\n", *(float*)vptr); //output: 3.14
}
//void pointers are used when we dont know the data type of the memory
//malloc returns void pointer 