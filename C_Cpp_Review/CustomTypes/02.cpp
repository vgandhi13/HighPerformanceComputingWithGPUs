// #include <stdio.h>
#include <iostream>

using namespace std;

typedef struct {
    float x;
    float y;
} Point;

int main() {
    Point p = {1.1, 2.5};
    printf("Size of Point: %zu\n", sizeof(Point));
}
//g++ 02.cpp -o 02.exe