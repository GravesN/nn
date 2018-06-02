#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::string fileAddress)
{
    //crée le réseau de neurones à partir de l'addresse du fichier
    std::ifstream file(fileAddress, std::ios::in); //ouverture du fichier en lecture
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
    m_nbLayer=nbLayer;
    m_layer=new Eigen::MatrixXd[m_nbLayer];
    m_weight=new Eigen::MatrixXd[m_nbLayer];
    m_bias=new Eigen::VectorXd[m_nbLayer]; //initilisation du réseau en créant des matrices vides
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

    for(int i=1;i<m_nbLayer;i++)
    {
        m_weight[i].setRandom();
        m_bias[i].setRandom();
        m_weight[i]/=10;
        m_bias[i]/=10;
    }
}

Eigen::MatrixXd const& NeuralNetwork::use(Eigen::MatrixXd const&input)
{
    std::cout << "problème pour l'input" << std::endl;
    std::cout << input << std::endl;
    std::cout << "passe ici" << std::endl;
    std::cout << m_layer[0].cols() << " " << input.cols() << std::endl;
    std::cout << m_layer[0].rows() << " " << input.rows() << std::endl;
    //std::cout << m_layer[0] << std::endl;
    //for (int i{0}; i< input.cols(); i++)
    //{
       // std::cout << m_layer[0](0,i) << std::endl;
        m_layer[0]=input;
       // std::cout << m_layer[0](0,i) << std::endl;
    //}
         //ok, l'erreur se fait ici, les vecteurs font bien la même taille. Le problème, c'est probablement le fait de copier une matrice constante
    std::cout << "problème pour le calcul" << std::endl;
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

