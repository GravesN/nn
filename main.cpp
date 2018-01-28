#include <iostream>
#include "fonctions.h"

int main()
{
    int *i{0};
    int u{1};
    i= &u;
    i=new int(3);
    delete i;
    return 0;
}
