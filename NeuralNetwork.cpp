#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::string fileAdress)
{

}

NeuralNetwork::NeuralNetwork(int nbLayer,int *nbNeuron,ActFunction *actFunction,int nbDataParCalcul)
{
    m_nbLayer=nbLayer;

    m_layer=new Eigen::MatrixXd[m_nbLayer];
    m_weight=new Eigen::MatrixXd[m_nbLayer];
    m_bias=new Eigen::VectorXd[m_nbLayer];

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

void NeuralNetwork::initvalue()
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

Eigen::MatrixXd NeuralNetwork::use(Eigen::MatrixXd input)
{
    m_layer[0]=input;
    for(int i{1};i<m_nbLayer;i++)
    {
        calculLayer(i);
    }

    return m_layer[m_nbLayer-1];
}

inline void NeuralNetwork::calculLayer(int number)
{
   m_layer[number]=m_actFunction[number]((m_weight[number]*m_layer[number-1]).colwise()+m_bias[number]);
}





