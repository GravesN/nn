#ifndef APRENTISSAGE_H
#define APRENTISSAGE_H

#include "NeuralNetwork.h"


class Aprentissage
{

    public:
        Aprentissage(std::string trucPourData);//je sais pas encore quoi
        ~Aprentissage();

        void learn();
        double validation();
        double test();

    private:

        void createNeuralNetwork();
        void setParameter();
        inline void feedForward();
        inline void calculOutputError();
        inline void backpropagation();
        inline void gradientDescend();

        int m_nbNetwork;
        int m_nbEpoch;//peut utilier un arret si plus de progression

        double *m_learningRate=0;//plus rapide pour les couche profondent car apprennent moin vite
        int m_miniBatchSize=1;//peut changer au cours du temp de + en+ gros pour aprendre vite au debut puis mieux converger

        double m_lambdaL1=0;
        double m_lambdaL2=0;
        //dropout
        //double m_momentumCoeff=0;

        ActFunction *m_actFunction=0;
        CostFunction *m_costFunction=0;

        NeuralNetwork* m_neuralNetwork=0;

        Database m_Data;

        int m_nbTrainingExemple;
        int m_nbLayer;
        int *nbNeuron=0;

        Eigen::MatrixXd *m_error=0;

};

#endif // APRENTISSAGE_H
