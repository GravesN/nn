#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Database.h"
#include <functional>
#include <random>
#include "fonctions.h"

class Aprentissage;
class NeuralNetwork
{
    friend class Aprentissage;

public:
    NeuralNetwork(std::string fileAddress);//construit un r�seau � partir d'un fichier (d�ja entrain� � priori)
    NeuralNetwork(int nbLayer,int *nbNeuron,ActFunction **actFunction,int nbDataParCalcul=1);//construit un r�seau � partir des donn�es fournies (non entrain� � priori)
    ~NeuralNetwork();
    Eigen::MatrixXd const& use(Eigen::MatrixXd const&input);
    SaveNeuralNetwork(std::string fileAddress,int nbLayer, ActFunction actFunction,Eigen::MatrixXd layer,Eigen::MatrixXd weight, Eigen::MatrixXd bias);
    // fonction pour enregistrer le r�seau, mais m�me probl�me, comprend-il que ce sont des matrices de matrices ?
private:

    void initvalue();
    void calcul();
    inline void calculLayer(int number);


    int m_nbLayer;

    Eigen::MatrixXd *m_layer{0};
    Eigen::MatrixXd *m_weight{0};
    Eigen::VectorXd *m_bias{0};

    ActFunction **m_actFunction{0};
};

#endif // NEURALNETWORK_H
