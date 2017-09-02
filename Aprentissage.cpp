#include "Aprentissage.h"

Aprentissage::Aprentissage(std::string trucPourData):m_Data{trucPourData}
{

}

Aprentissage::~Aprentissage()
{

}

void Aprentissage::learn()
{
    for(int i{0};i<m_nbNetwork;i++)
    {
        createNeuralNetwork();
        for(int j{0};j<m_nbEpoch;j++)
        {
            for(int k{0};k<m_nbTrainingExemple;k+=m_miniBatchSize)
            {
            //    m_learningData->loadInput(m_neuralNetwork->m_layer[0]);
                feedForward();
                calculOutputError();
                backpropagation();
                gradientDescend();
            }
        }
    }
}

void Aprentissage::createNeuralNetwork()
{

}

void Aprentissage::feedForward()
{
    for(int i{1};i<m_nbLayer;i++)
    {
        m_error[i]=(m_neuralNetwork->m_weight[i]*m_neuralNetwork->m_layer[i-1]).colwise()+m_neuralNetwork->m_bias[i];
        m_neuralNetwork->m_layer[i]=m_actFunction[i](m_error[i]);
    }
}

void Aprentissage::calculOutputError()
{

}

void Aprentissage::backpropagation()
{
    for(int i{m_nbLayer-2};i>0;i--)
    {
         m_error[i]=m_actFunction->prime(m_error[i]).cwiseProduct(m_neuralNetwork->m_weight[i+1].transpose()*m_error[i+1]);
    }
}

void Aprentissage::gradientDescend()
{
    for(int i{m_nbLayer-1};i>0;i--)
    {
        m_neuralNetwork->m_bias[i]-=(m_learningRate[i]/m_miniBatchSize)*m_error[i].rowwise().sum();
        m_neuralNetwork->m_weight[i]-=(m_learningRate[i]/m_miniBatchSize)*(m_error[i]*m_neuralNetwork->m_layer[i-1])//base
                                        +(m_learningRate[i]*m_lambdaL2/m_nbTrainingExemple)*m_neuralNetwork->m_weight[i]//L2
                                        +(m_learningRate[i]*m_lambdaL1/m_nbTrainingExemple)*m_neuralNetwork->m_weight[i].cwiseQuotient(m_neuralNetwork->m_weight[i].cwiseAbs());//L1
    }
}






