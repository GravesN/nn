#include "NeuralNetwork.h"
#include <fstream>

NeuralNetwork::NeuralNetwork(std::string fileAddress)
{
    //crée le réseau de neurones à partir de l'addresse du fichier
    std::ifstream file(fileAddress, std::ios::in); //ouverture du fichier en lecture
    if (file)
    {
        file>>m_nbLayer;//fonction d'activation ?

        file.close()
    }
     else
        std::cerr << "fichier non ouvert" << std::endl;
}

NeuralNetwork::NeuralNetwork(int nbLayer,int *nbNeuron,ActFunction **actFunction,int nbDataParCalcul) //les 0 dans la déclaration des pointeurs ?(cf OpenClassrooms)
{
    m_nbLayer=nbLayer;
            //le fait de faire tous les calculs en même temps n'aide clairement pas à la compréhension du code, j'ai un peu de mal à gérer les types.
    m_layer=new Eigen::MatrixXd[m_nbLayer];
    m_weight=new Eigen::MatrixXd[m_nbLayer];
    m_bias=new Eigen::VectorXd[m_nbLayer]; //initilisation du réseau en créant des matrices vides

    m_layer[0]=Eigen::MatrixXd::Zero(nbNeuron[0],nbDataParCalcul);
    m_weight[0]=Eigen::MatrixXd::Zero(0,0);
    m_bias[0]=Eigen::MatrixXd::Zero(0,0);
    for(int i{1};i<m_nbLayer;i++)
    {
        m_layer[i]=Eigen::MatrixXd::Zero(nbNeuron[i],nbDataParCalcul);
        m_weight[i]=Eigen::MatrixXd::Zero(nbNeuron[i],nbNeuron[i-1]);
        m_bias[i]=Eigen::MatrixXd::Zero(nbNeuron[i],1);
    }
    initvalue();
    m_actFunction=actFunction;
}

NeuralNetwork::~NeuralNetwork()
{
    delete[] m_layer;
    delete[] m_weight;
    delete[] m_bias;
}

NeuralNetwork::SaveNeuralNetwork(std::string fileAddress)
{   // écriture dans le fichier
    std::ofstream file(fileAddress, std::ios::out|std::ios::trunc)
    if(file)
    {   fichier << m_nbLayer << m_actFunction << std::endl;
        fichier << m_layer << std::endl;
        fichier << m_weight << std::endl;
        fichier << m_bias << std::endl;
        fichier.close();
    }
    else
        std::cerr << "fichier non ouvert" << std::endl
}

void NeuralNetwork::initvalue() //à lire complètement pour comprendre le fonctionnement
{
    int i{1};
    auto init=[&,i](double x)
    {
        std::random_device rd;
        std::normal_distribution<double> distr(0,1./sqrt(m_layer[i-1].size()));
        return distr(rd);

    };

    for(i=1;i<m_nbLayer;i++)
    {
        m_weight[i]=m_weight[i].unaryExpr(init);
        m_bias[i]=m_bias[i].unaryExpr(init);
    }
}

Eigen::MatrixXd const& NeuralNetwork::use(Eigen::MatrixXd const&input)
{
    m_layer[0]=input;
    calcul();
    return m_layer[m_nbLayer-1];
}

inline void NeuralNetwork::calcul()
{
    for(int i{1};i<m_nbLayer;i++)
    {
        calculLayer(i);
    }
}

inline void NeuralNetwork::calculLayer(int number)
{
   m_layer[number]=(*m_actFunction[number])((m_weight[number]*m_layer[number-1]).colwise()+m_bias[number]);
}

