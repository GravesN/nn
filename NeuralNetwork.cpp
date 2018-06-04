#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::string fileAddress)
{
    /// crée le réseau de neurones à partir de l'adresse du fichier sous le format indiqué ///
    std::ifstream file(fileAddress, std::ios::in);
    m_saveAddress=fileAddress;
    if (file)
    {
        file>>m_nbLayer;
        int* nbNeuron=new int[m_nbLayer];
        m_layer=new Eigen::MatrixXd[m_nbLayer];
        m_weight=new Eigen::MatrixXd[m_nbLayer];
        m_bias=new Eigen::VectorXd[m_nbLayer];
        m_actFunction= new ActFunction*[m_nbLayer];
        m_actFunction[0]=0;
        for (int i{1};i<m_nbLayer;i++)
        {
            std::string nomActFunction;
            file>>nomActFunction;
            if(nomActFunction=="Sigmoid")
                m_actFunction[i] = new Sigmoid;
            if(nomActFunction=="SoftMax")
                m_actFunction[i] = new SoftMax;
            if(nomActFunction=="Tanh")
                m_actFunction[i] = new Tanh;
            if(nomActFunction=="UpTanh")
                m_actFunction[i] = new UpTanh;
            if(nomActFunction=="ReLU")
                m_actFunction[i] = new ReLU;
        }

        for(int i{0};i<m_nbLayer;i++)
        {
            file>>nbNeuron[i];
        }
        m_layer[0]=Eigen::MatrixXd::Zero(nbNeuron[0],1);
        m_weight[0]=Eigen::MatrixXd::Zero(1,1);
        m_bias[0]=Eigen::MatrixXd::Zero(1,1);
        for(int i{1};i<m_nbLayer;i++)
        {
            m_layer[i]=Eigen::MatrixXd::Zero(nbNeuron[i],1);
            m_weight[i]=Eigen::MatrixXd::Zero(nbNeuron[i],nbNeuron[i-1]);
            m_bias[i]=Eigen::MatrixXd::Zero(nbNeuron[i],1);
        }
        for(int i{1};i<m_nbLayer;i++)
            for(int j{1};j<nbNeuron[i];j++)
                for(int k{1};k<nbNeuron[i-1];k++)
                    file>>m_weight[i](j,k);
        for(int i{1};i<m_nbLayer;i++)
            for(int j{1};j<nbNeuron[i];j++)
                    file>>m_bias[i](j,1);

        file.close();
        delete[] nbNeuron;
    }
     else
        std::cerr << "fichier non ouvert" << std::endl;
}

NeuralNetwork::NeuralNetwork(int nbLayer,int *nbNeuron,ActFunction const*const*actFunction,int nbDataParCalcul,bool save, std::string saveAddress) //les 0 dans la déclaration des pointeurs ?(cf OpenClassrooms)
{
    /// crée un nouveau réseau de neurones avec les caractéristiques données en argument ///
    m_nbLayer=nbLayer;
    m_layer=new Eigen::MatrixXd[m_nbLayer];
    m_weight=new Eigen::MatrixXd[m_nbLayer];
    m_bias=new Eigen::VectorXd[m_nbLayer];
    m_layer[0]=Eigen::MatrixXd::Zero(nbNeuron[0],nbDataParCalcul);
    m_weight[0]=Eigen::MatrixXd::Zero(1,1);
    m_bias[0]=Eigen::MatrixXd::Zero(1,1);
    for(int i{0};i<m_nbLayer;i++)
    {
        m_layer[i]=Eigen::MatrixXd::Zero(nbNeuron[i],nbDataParCalcul);
        m_weight[i]=Eigen::MatrixXd::Zero(nbNeuron[i],nbNeuron[i-1]);
        m_bias[i]=Eigen::MatrixXd::Zero(nbNeuron[i],1);
    }
    initvalue();
    m_actFunction= new ActFunction*[m_nbLayer];
    m_actFunction[0]=0;
    for (int i{1};i<m_nbLayer;i++)
    {
        std::string nomActFunction{actFunction[i]->nom()};
        if(nomActFunction=="Sigmoid")
            m_actFunction[i] = new Sigmoid;
        if(nomActFunction=="SoftMax")
            m_actFunction[i] = new SoftMax;
        if(nomActFunction=="Tanh")
            m_actFunction[i] = new Tanh;
        if(nomActFunction=="UpTanh")
            m_actFunction[i] = new UpTanh;
        if(nomActFunction=="ReLU")
            m_actFunction[i] = new ReLU;
    }
    m_save=save;
    m_saveAddress=saveAddress;
}

NeuralNetwork::~NeuralNetwork()
{
    /// destructeur du type NeuralNetwork ///
    if(m_save)
        saveNeuralNetwork();
    delete[] m_layer;
    delete[] m_weight;
    delete[] m_bias;
    for (int i{0};i<m_nbLayer;i++)
        delete m_actFunction[i];
    delete[] m_actFunction;

}

void NeuralNetwork::saveNeuralNetwork()
{
    /// sauvegarde le réseau de neurones sous le format indiqué ///
    std::ofstream file(m_saveAddress+".txt", std::ios::out | std::ios::trunc);
    if(file)
    {
        file << m_nbLayer << std::endl;
        for(int i{1};i<m_nbLayer;i++)
            file<<m_actFunction[i]->nom()<<" ";
        file<<std::endl;
        for(int i{0};i<m_nbLayer;i++)
            file<<m_layer[i].rows()<<" ";
        file<<std::endl<<std::endl;
        for(int i{1};i<m_nbLayer;i++)
            file << m_weight[i] << std::endl<<std::endl;
        file<<std::endl<<std::endl;
        for(int i{1};i<m_nbLayer;i++)
            file << m_bias[i] << std::endl<<std::endl;
        file.close();
    }
    else
        std::cerr << "fichier non ouvert" << std::endl;
}

void NeuralNetwork::initvalue()
{
    /// initialise les poids et les biais du réseau ///
    for(int i=1;i<m_nbLayer;i++)
    {
        m_weight[i].setRandom();
        m_bias[i].setRandom();
        m_weight[i]/=10;
        m_bias[i]/=10;
    }
}

Eigen::MatrixXd const& NeuralNetwork::use(Eigen::MatrixXd const&entree)
{
    /// calcule la sortie attendue pour l'entrée fournie ///
    m_layer[0]=entree;
    calcul();
    return m_layer[m_nbLayer-1];
}

inline void NeuralNetwork::calcul()
{
    /// calcule les valeurs dans toutes les couches, une fois l'entrée initialisée ///
    for(int i{1};i<m_nbLayer;i++)
    {
        calculLayer(i);
    }
}

inline void NeuralNetwork::calculLayer(int number)
{
    /// calcule les valeurs obtenues dans une couche ///
    m_layer[number]=(*m_actFunction[number])((m_weight[number]*m_layer[number-1]).colwise()+m_bias[number]);
}

