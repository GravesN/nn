#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <functional>
#include <random>
#include "fonctions.h"
#include <fstream>

class Apprentissage;
class NeuralNetwork
{
    friend class Apprentissage;

public:
    NeuralNetwork(std::string fileAddress);
    NeuralNetwork(int nbLayer,int *nbNeuron,ActFunction const*const*actFunction,int nbDataParCalcul=1,bool save=false, std::string saveAddress="");//construit un réseau à partir des données fournies (non entrainé à priori)
    ~NeuralNetwork();
    Eigen::MatrixXd const& use(Eigen::MatrixXd const&entree);

private:

    void initvalue();
    void calcul();
    inline void calculLayer(int number);
    void saveNeuralNetwork();

    int m_nbLayer;

    Eigen::MatrixXd *m_layer{0};
    Eigen::MatrixXd *m_weight{0};
    Eigen::VectorXd *m_bias{0};

    ActFunction **m_actFunction{0};

    bool m_save{false};
    std::string m_saveAddress;
};

#endif // NEURALNETWORK_H
