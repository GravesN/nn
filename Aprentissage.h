#ifndef APRENTISSAGE_H
#define APRENTISSAGE_H

#include "NeuralNetwork.h"
#include <thread>

class Aprentissage
{

public:
    Aprentissage();
    ~Aprentissage();
    void learn();

private:
    double test(int i);
    double validation(int i);
    void trainNetwork(int i);
    void createNeuralNetwork(int i);
    void setParameters(int i);
    double askParameter(std::string nom, double valeurMin=0, double valeurMax=0, double defaultValue=0, std::string textSup="");
    inline void feedForward(int i);
    inline void calculOutputError(int i);
    inline void backpropagation(int i);
    inline void gradientDescend(int i);

    int m_nbNetwork{1};
    int m_nbTread{1};
    int *m_nbEpoch{0};//peut utilier un arret si plus de progression
    std::thread* threads{0};

    double **m_learningRate{0};//plus rapide pour les couche profondent car apprennent moin vite
    int *m_miniBatchSize{0};//peut changer au cours du temps de + en+ gros pour aprendre vite au debut puis mieux converger

    double *m_lambdaL1{0};
    double *m_lambdaL2{0};
    //dropout
    //double *m_momentumCoeff{0};

    ActFunction **m_actFunction{0};
    CostFunction *m_costFunction{0};

    NeuralNetwork* m_neuralNetwork{0};

    Database *m_Data{0};

    int m_nbTrainingExemple;
    int *m_nbLayer{0};
    int **nbNeuron{0};

    Eigen::MatrixXd **m_error{0};

};

#endif // APRENTISSAGE_H
