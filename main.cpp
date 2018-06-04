#include "Aprentissage.h"

int main()
{
    srand(time(NULL));
    std::string parameterAddress{""};
    std::cout<<"adresse du fichier pour les parametres le cas echeant"<<std::endl;
    std::cin>>parameterAddress;
    std::ifstream flux(parameterAddress, std::ios::in);
    std::istream *fluxx(0);
    if(flux)
        fluxx=&flux;
    else
        fluxx=&std::cin;
    //bool reseauPresent;
    //std::cout << "Y a t-il un reseau de neurones pre-entraine ?" << std::endl;
    //std::cin >> reseauPresent;
    //std::string adresseReseau;
    //if (reseauPresent)
    //{
    //    std::cout << "adresse de ce reseau" << std::endl;
    //    std::cin >> adresseReseau;
    //}
    Apprentissage apprentissage{*fluxx};
    //if (not reseauPresent)
        apprentissage.learn();
    //else
       // apprentissage.construitReseau(adresseResesau);
    apprentissage.test();
    flux.close();
    return 0;
}



//aleat.txt -1000 3 5 -1 5 20 -1 5 20 -1 5 20 -1 0.05 0.8 -1 0.05 0.8 -1 0.05 0.8 -1 0.05 0.8 50 -1 1000 2000 0 0 1 2 5 5 5 5

// a xor.txt 9 3 3 5 -1 0.5 10 0.5 4 100000 0 0 1 2 5 5
