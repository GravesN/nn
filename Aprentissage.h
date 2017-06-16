#ifndef APRENTISSAGE_H
#define APRENTISSAGE_H

#include "NeuralNetwork.h"

class Aprentissage
{
    typedef double(*Fonction)(double);
    public:
        Aprentissage();
        ~Aprentissage();

        void learn();
        void backpropagation();



    private:

        void createNeuralNetwork();

        int m_nbINeurons;
        int m_nbONeurons;
        int m_nbHNeurons;

        double m_learningRate;
        int m_numberEpoch;

        int m_miniBatchSize;//=n pour pas la
        double m_lambdaL1=0;
        double m_lambdaL2=0;//egal les deux?
        //dropout
        double m_momentumCoeff=0;

        Fonction fonctionActivation=0;
        Fonction costFunction=0;

        NeuralNetwork* m_neuralNetwork=0;

        Database* m_learningData;
        Database* m_validationData;//peut etre plus haut

};

#endif // APRENTISSAGE_H
