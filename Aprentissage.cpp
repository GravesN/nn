#include "Aprentissage.h"

Aprentissage::Aprentissage()
{

}

Aprentissage::~Aprentissage()
{

}

void Aprentissage::learn()
{
    threads=new std::thread[m_nbTread];
    for(int i{0};i<m_nbNetwork/m_nbTread;i++)
    {
        createNeuralNetwork(i);
        threads[i]=std::thread(&Aprentissage::trainNetwork,this,i);
    }
}

double Aprentissage::test(int i)
{
    return 0;
}

double Aprentissage::validation(int i)
{
    return 0;
}

void Aprentissage::trainNetwork(int i)
{
    for(int j{0};j<m_nbEpoch[i];j++)
        {
            for(int k{0};k<m_nbTrainingExemple;k+=m_miniBatchSize[i])
            {
            //    m_learningData->loadInput(m_neuralNetwork->m_layer[0]);
                feedForward(i);
                calculOutputError(i);
                backpropagation(i);
                gradientDescend(i);
            }
        }
}

void Aprentissage::createNeuralNetwork(int i)
{

}

void Aprentissage::setParameters(int i)
{

}

double Aprentissage::askParameter(std::string nom, double valeurMin, double valeurMax, double defaultValue, std::string textSup)
{
    std::cout<<nom<<" :\n"<<"valeurMin: "<<valeurMin<<std::endl<<"valeurMax: "<<valeurMax<<std::endl<<"defaultValue: "<<defaultValue<<std::endl<<textSup<<std::endl<<"-66666 pour aléatoire\n";
    double valeur{defaultValue};
    std::cin>>valeur;
    if(valeur==-66666)
    {

    }
    return valeur;
}

void Aprentissage::feedForward(int i)
{
    for(int j{1};j<m_nbLayer[i];j++)
    {
        m_error[i][j]=(m_neuralNetwork[i].m_weight[j]*m_neuralNetwork[i].m_layer[j-1]).colwise()+m_neuralNetwork[i].m_bias[j];
        m_neuralNetwork[i].m_layer[j]=m_actFunction[i][j](m_error[i][j]);
    }
}

void Aprentissage::calculOutputError(int i)
{

}

void Aprentissage::backpropagation(int i)
{
    for(int j{m_nbLayer[i]-2};j>0;j--)
    {
         m_error[i][j]=m_actFunction[i][j].prime(m_error[i][j]).cwiseProduct(m_neuralNetwork[i].m_weight[j+1].transpose()*m_error[i][j+1]);
    }
}

void Aprentissage::gradientDescend(int i)
{
    for(int j{m_nbLayer[i]-1};j>0;j--)
    {
        m_neuralNetwork[i].m_bias[j]-=(m_learningRate[i][j]/m_miniBatchSize[i])*m_error[i][j].rowwise().sum();
        m_neuralNetwork[i].m_weight[j]-=(m_learningRate[i][j]/m_miniBatchSize[i])*(m_error[i][j]*m_neuralNetwork[i].m_layer[j-1])//base
                                        +(m_learningRate[i][j]*m_lambdaL2[i]/m_nbTrainingExemple)*m_neuralNetwork[i].m_weight[j]//L2
                                        +(m_learningRate[i][j]*m_lambdaL1[i]/m_nbTrainingExemple)*m_neuralNetwork[i].m_weight[j].cwiseQuotient(m_neuralNetwork[i].m_weight[j].cwiseAbs());//L1
    }
}






