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
        NeuralNetwork(std::string fileAdress);//construit un réseau à partir d'un fichier (déja entrainé à priori)
        NeuralNetwork(int nbLayer,int nbNeuron[],ActFunction *actFunction,int nbDataParCalcul=1);//construit un réseau aà partir des données fournie (non entrainé à priori)
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
