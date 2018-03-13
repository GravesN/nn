#include "Aprentissage.h"

Aprentissage::Aprentissage(std::string dataAddress, int nbNetwork, int nbTread): m_nbNetwork(nbNetwork),m_nbTread(nbTread)
{
    std::ifstream data(dataAddress, std::ios::in);

    if(data)
    {
        std::string type;
        data>>type; // sert à récupérer à partir du fichier jusqu'à un délimiteur
        // ici, on récupère donc tout pour le mettre dans une chaîne de caractères
        createDataBase(type, dataAddress);
        data.close();
    }
    else
        std::cerr << "data pas ouvert" << std::endl;

    m_nbTestExemple=m_data->getNbTestExemple();
}

Aprentissage::~Aprentissage()
{
    delete m_data;
}

void Aprentissage::learn()
{
    for(int j{0};j<m_nbNetwork/m_nbTread;j++)
    {
        std::thread threads[m_nbTread];
        setParameters();
        TrainSet trainSet[m_nbTread];
        for(int i{0};i<m_nbTread;i++)
        {
            trainSet[i].init(m_data);
            threads[i]=std::thread(&Aprentissage::TrainSet::trainNetwork,&trainSet[i]);
        }
        for(int i{0};i<m_nbTread;i++)
        {
            threads[i].join();
            int valid(trainSet[i].validation());
            if(valid>m_bestValidation)
            {
                m_bestValidation=valid;
                m_bestTrainSet=&trainSet[i];
            }
        }
    }
}

void Aprentissage::createDataBase(std::string type, std::string dataAddress)
{
// initialise m_data pour construire la base de données
    if(type=="bool")
        m_data=new DatabaseT<bool>{dataAddress};
    else if(type=="char")
        m_data=new DatabaseT<char>{dataAddress};
    else if(type=="unsignedChar")
        m_data=new DatabaseT<unsigned char>{dataAddress};
    else if(type=="shortInt")
        m_data=new DatabaseT<short int>{dataAddress};
    else if(type=="unsignedShortInt")
        m_data=new DatabaseT<unsigned short int>{dataAddress};
    else if(type=="int")
        m_data=new DatabaseT<int>{dataAddress};
    else if(type=="unsignedInt")
        m_data=new DatabaseT<unsigned int>{dataAddress};
    else if(type=="longInt")
        m_data=new DatabaseT<long int>{dataAddress};
    else if(type=="unsignedLongInt")
        m_data=new DatabaseT<unsigned long int>{dataAddress};
    else if(type=="longLongInt")
        m_data=new DatabaseT<long long int>{dataAddress};
    else if(type=="unsignedLongLongInt")
        m_data=new DatabaseT<unsigned long long int>{dataAddress};
    else if(type=="float")
        m_data=new DatabaseT<float>{dataAddress};
    else if(type=="double")
        m_data=new DatabaseT<double>{dataAddress};
    else if(type=="longDouble")
        m_data=new DatabaseT<long double>{dataAddress};
}

void Aprentissage::setParameters()
{
    /*demande à l'utilsateur(std::cin) les valeurs ou si aleatoire
    de tous les attributs de TrainSet puis les fais passer aux
    trainSet par init pour faire passer le plus simple c'est surement
    de rajouter des attributs a Aprentissage et de les rajouters comme
    parametre a init. J'ai mis des commentaires pour des variables dans le .h
    (surtout pour dire pourquoi c'est des pointeurs ou quand c'est moins facile de savoir
     ce que c'est vu que je les nommes pas très bien)*/
}

double Aprentissage::test()
{
    /*renvoie un indicateur de la réussite d'exactitude
    ou valeur de la fonction de cout ou autre
    pour la validation j'ai fait avec le cout
    t'aura surement besoin de loadTestInput de DataBase qui
    met les donnée dans ses arguments et de resizeMiniBatch pour
    mettre les matrice a la taille pour le nombre d'exemple de test
    (loadTestInput les charges tous d'un coup tu peux modifier si
     tu veux faire autrement*/
}

Aprentissage::TrainSet::TrainSet(){}

Aprentissage::TrainSet::~TrainSet()
{
    delete[] m_nbNeuron;
    for(int i{0};i<m_nbLayer;i++)
        delete m_actFunction[i];
    delete[] m_actFunction;
    delete m_costFunction;
    delete[] m_learningRate;
    delete m_sortieAttendue;
    delete[] m_error;
    delete m_neuralNetwork;
}

void Aprentissage::TrainSet::init(Database const* data)
{
    m_data=data;
    m_nbTrainingExemple=m_data->getNbTrainingExemple();
    m_nbValidationExemple=m_data->getNbValidationExemple();

    //initialiser toutes les autres variables a partir des résultat de setParameters

    m_sortieAttendue=new Eigen::MatrixXd(m_nbNeuron[m_nbLayer-1],m_miniBatchSize);
    m_error=new Eigen::MatrixXd[m_nbLayer];
    m_neuralNetwork=new NeuralNetwork(m_nbLayer,m_nbNeuron,m_actFunction,m_miniBatchSize);
    for(int i{0};i<m_nbLayer;i++)
        m_error[i]=m_neuralNetwork->m_layer[i];
}

double Aprentissage::TrainSet::validation()
{
    resizeMiniBatch(*m_nbValidationExemple);
    m_data->loadValidationInput(m_neuralNetwork->m_layer[0],*m_sortieAttendue);
    m_neuralNetwork->calcul();
    return (*m_costFunction)(m_neuralNetwork->m_layer[m_nbLayer-1],*m_sortieAttendue);
}

void Aprentissage::TrainSet::trainNetwork()
{
    for(int j{0};j<m_nbEpoch;j++)
        {
            for(int k{0};k<*m_nbTrainingExemple;k+=m_miniBatchSize)
            {
                m_data->loadTrainingInput(m_neuralNetwork->m_layer[0],*m_sortieAttendue,k,m_miniBatchSize);
                feedForward();
                calculOutputError();
                backpropagation();
                gradientDescend();
            }
        }
}


void Aprentissage::TrainSet::feedForward()
{
    for(int j{1};j<m_nbLayer;j++)
    {
        m_error[j]=(m_neuralNetwork->m_weight[j]*m_neuralNetwork->m_layer[j-1]).colwise()+m_neuralNetwork->m_bias[j];
        m_neuralNetwork->m_layer[j]=(*m_actFunction[j])(m_error[j]);
    }
}

void Aprentissage::TrainSet::calculOutputError()
{
     //determine la valeur de m_error[m_nbLayer-1]
}

void Aprentissage::TrainSet::backpropagation()
{
    for(int j{m_nbLayer-2};j>0;j--)
    {
         m_error[j]=m_actFunction[j]->prime(m_error[j]).cwiseProduct(m_neuralNetwork->m_weight[j+1].transpose()*m_error[j+1]);
    }
}

void Aprentissage::TrainSet::gradientDescend()
{
    for(int j{m_nbLayer-1};j>0;j--)
    {
        m_neuralNetwork->m_bias[j]-=(m_learningRate[j]/m_miniBatchSize)*m_error[j].rowwise().sum();
        m_neuralNetwork->m_weight[j]-=(m_learningRate[j]/m_miniBatchSize)*(m_error[j]*m_neuralNetwork->m_layer[j-1])//base
                                        +(m_learningRate[j]*m_lambdaL2/ *m_nbTrainingExemple)*m_neuralNetwork->m_weight[j]//L2
                                        +(m_learningRate[j]*m_lambdaL1/ *m_nbTrainingExemple)*m_neuralNetwork->m_weight[j].cwiseQuotient(m_neuralNetwork->m_weight[j].cwiseAbs());//L1
    }
}

void Aprentissage::TrainSet::resizeMiniBatch(int miniBatchSize)
{
    m_miniBatchSize=miniBatchSize;
    for(int i{0};i<m_nbLayer;i++)
    {
        m_error[i].resize(m_error[i].rows(),m_miniBatchSize);
        m_neuralNetwork->m_layer[i].resize(m_neuralNetwork->m_layer[i].rows(),m_miniBatchSize);
    }
    m_sortieAttendue->resize(m_sortieAttendue->rows(),m_miniBatchSize);
}
