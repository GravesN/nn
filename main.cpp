#include "Aprentissage.h"

int main()
{
    srand(time(NULL));
    std::string parameterAddress{""};
    std::cout<<"fichier pour les parametres existe pas pour sans"<<std::endl;
    std::cin>>parameterAddress;
    std::ifstream flux(parameterAddress, std::ios::in);
    std::istream *fluxx(0);
    if(flux)
    {
        fluxx=&flux;
    }
    else
    {
        fluxx=&std::cin;
    }
    Apprentissage apprentissage{*fluxx};
    apprentissage.learn();
    flux.close();

    return 0;
}



//aleat.txt -1000 3 5 -1 5 20 -1 5 20 -1 5 20 -1 0.05 0.8 -1 0.05 0.8 -1 0.05 0.8 -1 0.05 0.8 50 -1 1000 2000 0 0 1 2 5 5 5 5

// a xor.txt 9 3 3 5 -1 0.5 10 0.5 4 100000 0 0 1 2 5 5
