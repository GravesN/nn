#ifndef APRENTISSAGE_H
#define APRENTISSAGE_H

#include "NeuralNetwork.h"
#include <thread>

class Aprentissage
{

public:
    Aprentissage(std::string dataAdress, int nbNetwork, int nbTread); //c'est quoi nbTread ? (thread : processus d'exécution)
    ~Aprentissage();
    void learn();
    double test();
private:
    void createDataBase(std::string type, std::string dataAddress);
    void setParameters();

    int m_nbNetwork{1};
    int m_nbTread{1};

    Database *m_data{0};//pointeur pour pouvoir utiliser le polymorphisme avec l'héritage // ?
    int const* m_nbTestExemple{0};//pointeur constant car c'est celui de m_data

    class TrainSet
    {
    public:
        TrainSet();
        ~TrainSet();
        void init(Database const* data);
        void trainNetwork();
        double validation();
    private:


        inline void feedForward();
        inline void calculOutputError();
        inline void backpropagation();
        inline void gradientDescend();

        void resizeMiniBatch(int miniBatchSize);//batch ?

        int m_nbLayer;
        int *m_nbNeuron{0};//un tableau de taille m_nbLayer

        CostFunction *m_costFunction{0};//pointeur pour pouvoir choisir la classe (qui correspond à la fonction)
        ActFunction **m_actFunction{0};//comme pour cos mais en plus c'est un tableau donc pointeur sur pointeur du coup il faut faire new[] puis new sur chaque element

        double *m_learningRate{0};//plus rapide pour les couches profondes car elles apprennent moins vite
        int m_miniBatchSize;//peut changer au cours du temps de + en+ gros pour apprendre vite au début puis mieux converger

        Eigen::MatrixXd *m_sortieAttendue{0};//là c'est un pointeur juste pour l'initialiser quand on veut ça pourrait ne pas l'être
        Eigen::MatrixXd *m_error{0};//un tableau de taille m_nbLayer l'erreur pour la couche i a l'indice i
        NeuralNetwork *m_neuralNetwork{0};//c'est un pointeur pour l'initialiser quand on veut mais c'est plus utile que pour m_sortieAttendue car NeuralNetwork n'a pas de constructeur par défaut

        int m_nbEpoch;//le nombre de parcours des exemples
        double m_lambdaL1;//facteur pour la régularisaton L1 comme si pas la pour =0
        double m_lambdaL2;//facteur pour la régularisaton L2 comme si pas la pour =0


        //les trois là sont des pointeurs et constant car ils sont identiques pour tout les TrainSet
        Database const* m_data{0};
        int const* m_nbTrainingExemple{0};
        int const* m_nbValidationExemple{0};


    };

    TrainSet *m_bestTrainSet{0};//pointeur vers le meilleur
    int m_bestValidation{0}; //ce que renvoie m_bestTrainSet->validation()

};

#endif // APRENTISSAGE_H
