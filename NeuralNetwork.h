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
        NeuralNetwork(std::string fileAdress);//construit un r�seau � partir d'un fichier (d�ja entrain� � priori)
        NeuralNetwork(int nbLayer,int nbNeuron[],ActFunction *actFunction,int nbDataParCalcul=1);//construit un r�seau a� partir des donn�es fournie (non entrain� � priori)
        ~NeuralNetwork();
        Eigen::MatrixXd use(Eigen::MatrixXd input);

    private:

        void initvalue();
        inline void calculLayer(int number);

        int m_nbLayer;

        Eigen::MatrixXd *m_layer=0;
        Eigen::MatrixXd *m_weight=0;
        Eigen::VectorXd *m_bias=0;

        ActFunction *m_actFunction=0;
};

#endif // NEURALNETWORK_H
