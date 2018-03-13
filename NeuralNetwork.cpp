#include "NeuralNetwork.h"
#include <fstream>
//using namespace std;// à quoi cela sert ?

NeuralNetwork::NeuralNetwork(std::string fileAddress)
{
    //crée le réseau de neurones à partir de l'addresse du fichier
    std::ifstream file(fileAddress, std::ios::in); //ouverture du fichier en lecture
    if (file)
    {
        //ici ça dépend du choix de représentation de base de données que l'on choisit, je sais pas si c'est vraiment judicieux par contre
        int nbLayer, nbDataParCalcul; //stocke-t-on la fonction d'activation ?
        Eigen::MatrixX m_layer, m_weight, m_bias; //est-ce-qu'il comprend bien qu'on a affaire à une matrice de matrices ou faut-il le préciser ?
        ActFunction m_actFunction;
        file>>nbLayer>>nbDataParCalcul>>m_actFunction>>m_layer>>m_weight>>m_bias;
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

    m_layer[0]=Eigen::MatrixXd::Zero(nbNeuron[0],nbDataParCalcul); //revoir le fonctionnement des pointeurs pour comprendre le rôle du 0
    m_weight[0]=Eigen::MatrixXd::Zero(0,0);
    m_bias[0]=Eigen::MatrixXd::Zero(0,0); //pour l'instant je ne comprend pas vraiment (remplissage de vecteurs par des matrices)
    for(int i{1};i<m_nbLayer;i++)
    {
        m_layer[i]=Eigen::MatrixXd::Zero(nbNeuron[i],nbDataParCalcul);
        m_weight[i]=Eigen::MatrixXd::Zero(nbNeuron[i],nbNeuron[i-1]);
        m_bias[i]=Eigen::MatrixXd::Zero(nbNeuron[i],1);// même problème que pour l'initialisation d'au-dessus (remplissage de vecteurs par des matrices)
    }
    initvalue();
    m_actFunction=actFunction;
    //la fonction ne renvoie rien, faut-il mettre un void ?
}

NeuralNetwork::~NeuralNetwork()
{
    delete[] m_layer;
    delete[] m_weight;
    delete[] m_bias;
}

NeuralNetwork::SaveNeuralNetwork(std::string fileAddress, int nbLayer, int nbDataParCalcul, Eigen::MatrixX m_layer, m_weight, m_bias, ActFunction m_actFunction)
{   // écriture dans le fichier
    std::ofstream file(fileAddress, std::ios::out|std::ios::trunc)
    if(file)
    {   fichier << nbLayer << nbDataParCalcul << m_actFunction << std::endl;
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

