#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Database.h"

class Aprentissage;
class NeuralNetwork
{
    friend class Aprentissage;

    typedef double(*Fonction)(double);

    public:
        NeuralNetwork();
        ~NeuralNetwork();
        double use(double T);


    private:

        const int m_nbINeurons=0;
        const int m_nbONeurons=0;
        const int m_nbHNeurons=0;

        double Tm_iNeurons;
        double Tm_oNeurons;
        double T2m_hNeurons;

        double T2m_oweight;
        double T3m_hweight;

        double T2m_obias;
        double T3m_hbias;

        Fonction fonctionActivation=0;
};

#endif // NEURALNETWORK_H
