#include "Aprentissage.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

Apprentissage::Apprentissage(std::istream &flux):m_flux(flux)
{
    std::string dataAddress;
    std::cout<<"dataAddress"<<std::endl;
    m_flux>>dataAddress;
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

    std::cout<<"nbNetwork <-100 pour infini"<<std::endl;
    m_flux>>m_nbNetwork;

    std::cout<<"nbThread"<<std::endl;
    m_flux>>m_nbThread;

    m_nbTestExemple=m_data->getNbTestExemple();
    m_saveAddress=dataAddress+"aa/";
    setParameters();
}

Apprentissage::~Apprentissage()
{
    delete m_data;
    for (int i{0};i<m_nbLayer[2];i++)
    {
        delete[] m_nbNeuron[i];
        delete[] m_learningRate[i];
        delete m_actFunction[i];
    }
    delete[] m_nbNeuron;
    delete[] m_learningRate;

    delete[] m_miniBatchSize;
    delete[] m_nbEpoch;
    delete[] m_lambdaL1;
    delete[] m_lambdaL2;

    delete m_costFunction;
    delete[] m_nbLayer;
    delete[] m_actFunction;
}

void Apprentissage::learn()
{
    std::thread threadStop(&Apprentissage::stop,this);
    for(int j{0};j!=m_nbNetwork/m_nbThread&&!m_stop;j++)
    {
        std::thread threads[m_nbThread];

        TrainSet trainSet[m_nbThread];
        for(int i{0};i<m_nbThread;i++)
        {
            trainSet[i].init(m_data, m_nbLayer, m_nbNeuron, m_learningRate, m_miniBatchSize, m_nbEpoch, m_lambdaL1, m_lambdaL2, m_costFunction, m_actFunction, &m_stop,m_save,j*m_nbThread+i,m_saveAddress);
            threads[i]=std::thread(&Apprentissage::TrainSet::trainNetwork,&trainSet[i]);
        }
        for(int i{0};i<m_nbThread;i++)
        {
            threads[i].join();
            std::cout << "join réalisé" << std::endl;
            double valid(trainSet[i].validation());
            std::cout << "arrivé à la validation" << std::endl;
            if(valid>m_bestValidation||m_bestTrainSet==0)//modifié ici
            {
                m_bestValidation=valid;
                if(m_bestTrainSet!=0)
                    m_bestTrainSet->setSave(m_save);
                m_bestTrainSet=&trainSet[i];
                m_bestTrainSet->setSave(true);
            }
            std::cout << "sortie de la validation" << std::endl;
        }

    }
    std::cout << "bien trouvé le meilleur trainSet" << std::endl;
    m_stop=true;
    test();
    threadStop.join();
}

void Apprentissage::createDataBase(std::string type, std::string dataAddress)
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

void Apprentissage::setParameters()
{
    choisir("m_nbLayer", m_nbLayer);
    m_nbNeuron = new int*[m_nbLayer[2]];
    m_nbNeuron[0]=new int[3];
    m_nbNeuron[m_nbLayer[2]-1]=new int[3];
    for (int i{0};i<3;i++)
    {
        m_nbNeuron[m_nbLayer[2]-1][i]=m_data->getOutputSize();
        m_nbNeuron[0][i]=m_data->getInputSize();
    }
    for (int i{1};i<m_nbLayer[2]-1;i++)
    {
        m_nbNeuron[i] = new int[3];
        std::cout<<"Initialisation de m_nbNeuron dans la couche eventuelle "<< i<<std::endl;
        choisir("m_nbNeuron", m_nbNeuron[i]);
    }
    m_learningRate = new double*[m_nbLayer[2]];
    m_learningRate[0]=0;
    for (int i{1};i<m_nbLayer[2];i++)
    {
        m_learningRate[i] = new double[3];
        std::cout<<"Initialisation de m_learningRate dans la couche eventuelle "<< i<<std::endl;
        choisir("m_learningRate", m_learningRate[i]);
    }
    choisir("m_miniBatchSize", m_miniBatchSize);
    choisir("m_nbEpoch", m_nbEpoch);
    choisir("m_lambdaL1", m_lambdaL1);
    choisir("m_lambdaL2", m_lambdaL2);
    std::cout<<"Faut il sauvegarder les parametres detailles, avec le meilleur validationScore ?"<<std::endl;
    m_flux>>m_save;
    std::cout<<"Faut il sauvegarder les parametres bruts ?"<<std::endl;
    m_flux>>m_save_brut;
    if(m_save_brut)
    {
        std::ofstream file(m_saveAddress+"parametres.txt", std::ios::out | std::ios::trunc);
        if(file)
        {
            file << m_data->nom() << std::endl;
            file << m_nbNetwork << std::endl;
            file << m_nbThread << std::endl;
            file << m_nbLayer[0] << std::endl;
            for(int i{1};i<m_nbLayer[2]-1;i++)
                file << m_nbNeuron[i][0] << " ";
            file << std::endl;
            for(int i{1};i<m_nbLayer[2];i++)
                file << m_learningRate[i][0] << " ";
            file<<std::endl;
            file << m_miniBatchSize[0] << std::endl;
            file << m_nbEpoch[0] << std::endl;
            file << m_lambdaL1[0] << " " << m_lambdaL2[0] << std::endl;
            file << m_save << std::endl;
            file << m_save_brut << std::endl;

            file.close();
        }
        else
            std::cerr << "fichier non ouvert" << std::endl;
    }
    choisirCostFunction();
    choisirActFunction();

}

void Apprentissage::stop()
{
    std::string st;
    while(!m_stop)
    {
        std::cout<<"taper stop pour stopper"<<std::endl;
        std::cin>>st;
        if(st=="stop")
            m_stop=true;
    }
}

double Apprentissage::TrainSet::evaluation(double *flottant)
{
    if(flottant[0]<0)//cas aléatoire
    {
        return rand()/(double) RAND_MAX  * (flottant[2]-flottant[1])+flottant[1];
    }
    else
        return flottant[0];
}

 int Apprentissage::TrainSet::evaluation(int *entier)
{
    if(entier[0]<0)//cas aléatoire
    {
        return rand()%(entier[2]-entier[1])+entier[1];
    }
    else
        return entier[0];
}

void Apprentissage::choisir(std::string texte, int *&entier)
{
    std::cout<<"Entrez "<< texte <<" - negatif pour aleatoire"<<std::endl;
    m_flux>>entier[0];
    if (entier[0]<0)
    {
        std::cout<<"Entrez l'intervalle dans le cas aleatoire, 2 entiers arbitraires sinon"<<std::endl;
        m_flux>>entier[1]>>entier[2];
    }
    else
    {
        entier[1] = entier[0];
        entier[2] = entier[0];
    }
}

void Apprentissage::choisir(std::string texte, double *&flottant)
{
    std::cout<<"Entrez "<< texte <<" - negatif pour aleatoire"<<std::endl;
    m_flux>>flottant[0];
    if (flottant[0]<0)
    {
        std::cout<<"Entrez l'intervalle dans le cas aleatoire, 2 flottants arbitraires sinon"<<std::endl;
        m_flux>>flottant[1]>>flottant[2];
    }
    else
    {
        flottant[1] = flottant[0];
        flottant[2] = flottant[0];
    }
}

void Apprentissage::choisirCostFunction()
{
    std::cout<<"Entrez la fonction de cout :"<< std::endl;
    std::cout<<"1 pour CrossEntropy, 2 pour Quadratic, 3 pour LogLikelihood"<<std::endl;//mettre un numéro pour chaque fonction de cout
    int idCostFunction;
    m_flux>>idCostFunction;
    if(m_save_brut)
    {
        std::ofstream file(m_saveAddress+"parametres.txt", std::ios::out | std::ios::app);
        file << idCostFunction <<std::endl;
        file.close();
    }
    if(idCostFunction==1)
        m_costFunction = new CrossEntropy;
    if(idCostFunction==2)
        m_costFunction = new Quadratic;
    if(idCostFunction==3)
        m_costFunction = new LogLikelihood;
}

void Apprentissage::choisirActFunction()
{
    m_actFunction = new ActFunction*[m_nbLayer[2]];
    m_actFunction[0]=0;
    std::cout<<"Entrez les "<<m_nbLayer[2]-1<<" fonctions d'activation eventuelles:"<< std::endl;
    std::cout<<"1 pour Sigmoid, 2 pour SoftMax, 3 pour Tanh, 4 pour UpTanh, 5 pour ReLU"<<std::endl;
    for (int i{1};i<m_nbLayer[2];i++)//revoir les indices
    {
        int idActFunction;
        m_flux>>idActFunction;
        if(m_save_brut)
            {
                std::ofstream file(m_saveAddress+"parametres.txt", std::ios::out | std::ios::app);
                file << idActFunction << " ";
                file.close();
            }
        if(idActFunction==1)
            m_actFunction[i] = new Sigmoid;
        if(idActFunction==2)
            m_actFunction[i] = new SoftMax;
        if(idActFunction==3)
            m_actFunction[i] = new Tanh;
        if(idActFunction==4)
            m_actFunction[i] = new UpTanh;
        if(idActFunction==5)
            m_actFunction[i] = new ReLU;
    }
}

double Apprentissage::test()
{
    //NeuralNetwork *m_neuralNetwork{0};
    //*m_neuralNetwork = NeuralNetwork(m_saveAddress+"neuralNetwork0.txt");//à modifier, pour faire avec le meilleur trainSet uniquement
    //std::cout << "problème lors de la copie" << std::endl;
    std::cout << "entree dans test" << std::endl;
    //m_bestTrainSet->resizeMiniBatch(*m_nbTestExemple);
    std::cout << "ok pour le resizeminiBatch" << std::endl;
    int const m_ouputSize = m_data->getOutputSize();
    std::cout << m_ouputSize << std::endl;
    const Eigen::MatrixXd m_sortieVraie = m_data->getResultTestOutput();
    const Eigen::MatrixXd m_entree = m_data->getTestInput();
    std::cout << "voici l'entree telle que recuperee : " << m_entree << std::endl;
    //Eigen::MatrixXd m_resultTestData(m_sortie.rows(), m_sortie.cols());
    //Eigen::MatrixXd m_TestInput(m)
    //std::cout << m_data->getResultTestOutput() << std::endl;

    //copie(m_resultTestData, (m_data->getResultTestOutput()));
    std::cout << "problème lors du calcul de la sortie" << std::endl;
    std::cout << m_bestTrainSet->m_neuralNetwork->m_layer[0] << std::endl;
    const Eigen::MatrixXd m_sortieCalculee = m_bestTrainSet->m_neuralNetwork->use(m_entree);//c'est probablement l'entrée qui n'est pas de la bonne taille
    //double valid{(*m_costFunction)(m_neuralNetwork->m_layer[m_nbLayer-1],*m_sortieAttendue)};

    Eigen::MatrixXd probasTest(2, *m_nbTestExemple);
    std::cout << "problème pour le calcul de l'erreur" << std::endl;
    return calculErreurTest(probasTest, m_sortieVraie, m_sortieCalculee);
    /*renvoie un indicateur de la réussite d'exactitude
    ou valeur de la fonction de cout t'auras surement besoin de loadTestInput de DataBase qui
    met les données dans ses arguments et de resizeMiniBatch pour
    mettre les matrice a la taille pour le nombre d'exemple de test
    (loadTestInput les charges tous d'un coup tu peux modifier si tu veux faire autrement*/
    //delete m_neuralNetwork;
}

Apprentissage::TrainSet::TrainSet(){}

double Apprentissage::calculErreurTest(Eigen::MatrixXd probas, Eigen::MatrixXd m_sortieVraie, Eigen::MatrixXd m_sortieCalculee)
{
    //std::cout << "problème dans m_ouputSize" << std::endl;
    int m_outputSize = m_sortieVraie.rows();
    int nbExemples = m_sortieVraie.cols();
    for (int id{0} ; id < nbExemples; id++)
    {
        double true_negative{0};
        double true_positive{0};
        double false_negative{0};
        double false_positive{0};
        for(int i{0}; i< m_outputSize; i++)//avant : m_outputSize * m_nbTestExemple
        {
            if(m_sortieVraie(i, id) == 0)//normalement c'est un entier mais il faut voir si ça ne cause aucun problème
            {
                if(m_sortieCalculee(i, id) <= 0.5)
                    true_negative +=1;
                else
                    false_positive +=1;
            }
            else
            {
                if(m_sortieCalculee(i, id) > 0.5)
                    true_positive +=1;
                else
                    false_negative +=1;
            }
        }
        probas(0,id) = true_positive / (true_positive + false_negative); //true_positive_rate
        probas(1,id) = false_positive / (true_negative + false_positive); //false_discovery_rate
    }
    //std::cout << "problème dans le tri rapide" << std::endl;
    m_bestTrainSet->triRapide(probas, 0, nbExemples -1);//le tri est sur place
    //calcul de l'intégrale de la receiver operating characteristic
    double I{0.0};
    //std::cout << "problème dans le calcul de l'intégrale" << std::endl;
    for(int k{0}; k< nbExemples-1; k++)
    {
        I  += (probas(0,k+1) - probas(0,k))*(probas(1,k+1) + probas(1,k))/2;
    }
    return I;
}

Apprentissage::TrainSet::~TrainSet()
{
    if(m_save)
        save();
    delete[] m_nbNeuron;
    delete[] m_learningRate;
    delete m_sortieAttendue;
    delete[] m_error;
    delete m_neuralNetwork;
}

void Apprentissage::TrainSet::init(Database const* data, int *nbLayer, int **nbNeuron, double **learningRate, int *miniBatchSize, int *nbEpoch, double *lambdaL1, double *lambdaL2, CostFunction *costFunction, ActFunction **actFunction,bool *stop,bool save, int id, std::string saveAddress)
{
    m_stop=stop;
    m_save=save;
    m_id=id;
    m_saveAddress=saveAddress+"trainSet"+intToString(id);
    m_data=data;
    m_nbTrainingExemple=m_data->getNbTrainingExemple();
    m_nbValidationExemple=m_data->getNbValidationExemple();
    m_nbLayer = evaluation(nbLayer);
    m_nbNeuron = new int[m_nbLayer];
    for (int i{0}; i<m_nbLayer; i++)
        m_nbNeuron[i] = evaluation(nbNeuron[i]);
    m_learningRate = new double[m_nbLayer];
    for (int i{1}; i<m_nbLayer; i++)
        m_learningRate[i] = evaluation(learningRate[i]);
    m_miniBatchSize = evaluation(miniBatchSize);
    m_nbEpoch = evaluation(nbEpoch);
    m_lambdaL1 = evaluation(lambdaL1);
    m_lambdaL2 = evaluation(lambdaL2);
    m_costFunction = costFunction;
    m_actFunction = &actFunction[nbLayer[2]-m_nbLayer];
    m_sortieAttendue=new Eigen::MatrixXd(m_nbNeuron[m_nbLayer-1],m_miniBatchSize);
    m_error=new Eigen::MatrixXd[m_nbLayer];
    m_neuralNetwork=new NeuralNetwork(m_nbLayer,m_nbNeuron,m_actFunction,m_miniBatchSize,m_save,saveAddress+"neuralNetwork"+intToString(id));
    for(int i{0};i<m_nbLayer;i++)
        m_error[i]=m_neuralNetwork->m_layer[i];

}

double Apprentissage::TrainSet::validation()
{

    int miniBatchSize=m_miniBatchSize;
    resizeMiniBatch(*m_nbValidationExemple);
    m_data->loadValidationInput(m_neuralNetwork->m_layer[0],*m_sortieAttendue);
    m_neuralNetwork->calcul();
    //double valid{(*m_costFunction)(m_neuralNetwork->m_layer[m_nbLayer-1],*m_sortieAttendue)};
    Eigen::MatrixXd probasValidation(2, *m_nbValidationExemple);
    //std::cout << "problème dans le calcul de l'erreur" << std::endl;
    double valid = calculErreur(probasValidation, *m_nbValidationExemple);
    //std::cout << "problème plus tard" << std::endl;
    m_validationScore=valid;
    resizeMiniBatch(miniBatchSize);
    //std::cout << "sortie de valid" << std::endl;
    return valid;
}

void  Apprentissage::TrainSet::setSave(bool save)
{
    m_save=save;
    m_neuralNetwork->m_save=save;
}

void Apprentissage::TrainSet::trainNetwork()
{
    for(int j{0};j<m_nbEpoch&&!*m_stop&&!earlyStopping();j++)
    {
        for(int k{0};k<*m_nbTrainingExemple;k+=m_miniBatchSize)
        {
            m_data->loadTrainingInput(m_neuralNetwork->m_layer[0],*m_sortieAttendue,k,m_miniBatchSize);
            feedForward();
            calculOutputError();
            backpropagation();
            gradientDescend();
        }
        if (j%20==0)
            std::cout<<"set "<<m_id<<" epoch "<<j<<std::endl;
    }
    //std::cout << "sortie de trainNetwork" << std::endl;
}

void Apprentissage::TrainSet::feedForward()
{
    for(int j{1};j<m_nbLayer;j++)
    {
        m_error[j]=(m_neuralNetwork->m_weight[j]*m_neuralNetwork->m_layer[j-1]).colwise()+m_neuralNetwork->m_bias[j];
        m_neuralNetwork->m_layer[j]=(*m_actFunction[j])(m_error[j]);
    }
}

void Apprentissage::TrainSet::calculOutputError()
{
    m_error[m_nbLayer-1]=m_actFunction[m_nbLayer-1]->prime(m_error[m_nbLayer-1]).cwiseProduct(m_costFunction->gradient(m_neuralNetwork->m_layer[m_nbLayer-1],*m_sortieAttendue));
}

void Apprentissage::TrainSet::backpropagation()
{
    for(int j{m_nbLayer-2};j>0;j--)
    {
         m_error[j]=m_actFunction[j]->prime(m_error[j]).cwiseProduct(m_neuralNetwork->m_weight[j+1].transpose()*m_error[j+1]);
    }
}

void Apprentissage::TrainSet::gradientDescend()
{
    for(int j{m_nbLayer-1};j>0;j--)
    {
        m_neuralNetwork->m_bias[j]-=(m_learningRate[j]/m_miniBatchSize)*m_error[j].rowwise().sum();
        m_neuralNetwork->m_weight[j]-=(m_learningRate[j]/m_miniBatchSize)*(m_error[j]*((m_neuralNetwork->m_layer[j-1]).transpose()))//base
                                        +(m_learningRate[j]*m_lambdaL2/ *m_nbTrainingExemple)*m_neuralNetwork->m_weight[j]//L2
                                        +(m_learningRate[j]*m_lambdaL1/ *m_nbTrainingExemple)*m_neuralNetwork->m_weight[j].cwiseQuotient(m_neuralNetwork->m_weight[j].cwiseAbs());//L1
    }
}

void Apprentissage::TrainSet::resizeMiniBatch(int miniBatchSize)
{
    m_miniBatchSize=miniBatchSize;
    std::cout << "entree dans la boucle " << std::endl;
    for(int i{0};i<m_nbLayer;i++)
    {
        m_error[i].resize(m_error[i].rows(),m_miniBatchSize);
        std::cout << "milieu de la boucle " << std::endl;
        m_neuralNetwork->m_layer[i].resize(m_neuralNetwork->m_layer[i].rows(),m_miniBatchSize);
    }
    m_sortieAttendue->resize(m_sortieAttendue->rows(),m_miniBatchSize);
}

bool Apprentissage::TrainSet::earlyStopping()
{
    return 0;
}

void Apprentissage::TrainSet::save()
{
    std::ofstream file(m_saveAddress+".txt", std::ios::out | std::ios::trunc);
    if(file)
    {
        file << m_data->nom() << std::endl;
        file << m_nbLayer << std::endl;
        file << m_costFunction->nom() << std::endl;
        for(int i{1};i<m_nbLayer;i++)
            file<<m_actFunction[i]->nom()<<" ";
        file<<std::endl;
        for(int i{0};i<m_nbLayer;i++)
            file<<m_nbNeuron[i]<<" ";
        file<<std::endl;
        for(int i{0};i<m_nbLayer;i++)
            file << m_learningRate[i] << " ";
        file<<std::endl;
        file << m_miniBatchSize << std::endl;
        file << m_nbEpoch << std::endl;
        file << m_lambdaL1 << std::endl;
        file << m_lambdaL2 << std::endl;
        file << m_validationScore << std::endl;

        file.close();
    }
    else
        std::cerr << "fichier non ouvert" << std::endl;
}

double Apprentissage::TrainSet::calculErreur(Eigen::MatrixXd probas, int nbExemples)
{
    //std::cout << "problème dans m_ouputSize" << std::endl;
    int m_outputSize = (*m_sortieAttendue).rows();
    for (int id{0} ; id < nbExemples; id++)
    {
        double true_negative{0};
        double true_positive{0};
        double false_negative{0};
        double false_positive{0};
        for(int i{0}; i< m_outputSize; i++)//avant : m_outputSize * m_nbTestExemple
        {
            if((*m_sortieAttendue)(i, id) == 0)//normalement c'est un entier mais il faut voir si ça ne cause aucun problème
            {
                if((m_neuralNetwork->m_layer[m_nbLayer-1])(i, id) <= 0.5)
                    true_negative +=1;
                else
                    false_positive +=1;
            }
            else
            {
                if((m_neuralNetwork->m_layer[m_nbLayer-1])(i, id) > 0.5)
                    true_positive +=1;
                else
                    false_negative +=1;
            }
        }
        probas(0,id) = true_positive / (true_positive + false_negative); //true_positive_rate
        probas(1,id) = false_positive / (true_negative + false_positive); //false_discovery_rate
    }
    //std::cout << "problème dans le tri rapide" << std::endl;
    triRapide(probas, 0, nbExemples -1);//le tri est sur place
    //calcul de l'intégrale de la receiver operating characteristic
    double I{0.0};
    //std::cout << "problème dans le calcul de l'intégrale" << std::endl;
    for(int k{0}; k< nbExemples-1; k++)
    {
        I  += (probas(0,k+1) - probas(0,k))*(probas(1,k+1) + probas(1,k))/2;
    }
    return I+(1.0 - probas(0, nbExemples))*(1.0+probas(1,nbExemples))/2;
}

void Apprentissage::TrainSet::triRapide(Eigen::MatrixXd t, int i, int j)
{
    if(i+1 < j)
    {
        int a = segmente(t, i, j);
        triRapide(t, i, a);
        triRapide(t, a+1, j);
    }
}

int Apprentissage::TrainSet::segmente(Eigen::MatrixXd t, int i, int j)
{
    double pivot = t(0,j-1);
    int a = i;
    for(int b{i}; b<j-1 ; b++)
    {
        if(t(0,b)<pivot)
        {
            t(0,a), t(0,b), t(1,a), t(1,b) = t(0,b), t(0,a), t(1,b), t(1,a);
            a += 1;
        }
         t(0,a), t(0,j-1), t(1,a), t(1,j-1) = t(0,j-1), t(0,a), t(1,j-1), t(1,a);
    }
    return a;
}

void Apprentissage::copie(Eigen::MatrixXd matrice, const Eigen::MatrixXd matConst)
{
    for (int i{0}; i< matConst.rows(); i++)
    {
        for (int j{0}; j< matConst.cols(); j++)
            matrice(i,j) = matConst(i,j);
    }
}
