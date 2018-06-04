#ifndef APRENTISSAGE_H
#define APRENTISSAGE_H

#include "NeuralNetwork.h"
#include <thread>
#include "Database.h"


class Apprentissage
{

public:
    Apprentissage(std::istream &flux);
    ~Apprentissage();
    void learn();
    void test();
    //void construitReseau(std::string adresseReseau, std::string dataAddress);
private:
    void calculErreurTest(Eigen::MatrixXd probas, Eigen::MatrixXd m_sortieVraie, Eigen::MatrixXd m_sortieCalculee);
    void createDataBase(std::string type, std::string dataAddress);
    void setParameters();

    void choisirCostFunction();
    void choisirActFunction();
    void choisir(std::string texte, int *&entier);
    void choisir(std::string texte, double *&flottant);

    void stop();
    bool m_stop{false};
    bool m_save_brut{false};

    void copie(Eigen::MatrixXd matrice, const Eigen::MatrixXd matConst);

    std::istream &m_flux;
    int m_nbNetwork{1};
    int m_nbThread{1};

    Database *m_data{0};//pointeur pour pouvoir utiliser le polymorphisme avec l'héritage // ?
    int const* m_nbTestExemple{0};//pointeur constant car c'est celui de m_data

    int *m_nbLayer = new int[3];
    int **m_nbNeuron{0};
    double **m_learningRate{0};
    int *m_miniBatchSize = new int[3];
    int *m_nbEpoch = new int[3];
    double *m_lambdaL1 = new double[3];
    double *m_lambdaL2 = new double[3];
    bool m_save;
    std::string m_saveAddress;

    CostFunction *m_costFunction{0};
    ActFunction **m_actFunction{0};



    class TrainSet
    {
    public:
        TrainSet();
        ~TrainSet();
        void init(Database const* data, int *nbLayer, int **nbNeuron, double **learningRate, int *miniBatchSize, int *nbEpoch, double *lambdaL1, double *lambdaL2, CostFunction *costFunction, ActFunction **actFunction,bool *stop,bool save, int id,std::string saveAddress);
        void trainNetwork();
        double validation();
        void setSave(bool save);
        bool m_save;
        void save();
        NeuralNetwork *m_neuralNetwork{0};

        double calculErreur(Eigen::MatrixXd probas, int nbExemples);
        int segmente(Eigen::MatrixXd t, int i, int j);
        void triRapide(Eigen::MatrixXd t, int i, int j);

        CostFunction const*m_costFunction{0};//pointeur pour pouvoir choisir la classe (qui correspond à la fonction)
        ActFunction const*const*m_actFunction{0};//comme pour cos mais en plus c'est un tableau donc pointeur sur pointeur du coup il faut faire new[] puis new sur chaque element

        double *m_learningRate{0};//plus rapide pour les couches profondes car elles apprennent moins vite
        int m_miniBatchSize;//peut changer au cours du temps de + en+ gros pour apprendre vite au début puis mieux converger

        void resizeMiniBatch(int miniBatchSize);
        int m_nbLayer;
        int *m_nbNeuron{0};
         int m_nbEpoch;//le nombre de parcours des exemples
        double m_lambdaL1;//facteur pour la régularisaton L1 comme si pas la pour =0
        double m_lambdaL2;//facteur pour la régularisaton L2 comme si pas la pour =0

        double m_validationScore{0};

        //les trois là sont des pointeurs et constant car ils sont identiques pour tout les TrainSet
        Database const* m_data{0};
        int const* m_nbTrainingExemple{0};
        int const* m_nbValidationExemple{0};
        Eigen::MatrixXd *m_error{0};
        Eigen::MatrixXd *m_sortieAttendue{0};
        bool const* m_stop{0};
        std::string m_saveAddress;

    private:

        int evaluation(int *entier);
        double evaluation(double *flottant);

        inline void feedForward();
        inline void calculOutputError();
        inline void backpropagation();
        inline void gradientDescend();

        bool earlyStopping();

        int m_id;

    };

    TrainSet *m_bestTrainSet{0};//pointeur vers le meilleur
    int m_bestValidation{0}; //ce que renvoie m_bestTrainSet->validation()

};

#endif // APRENTISSAGE_H
