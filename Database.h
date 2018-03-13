#ifndef DATABASE_H
#define DATABASE_H

#include <Eigen/Dense>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

class Database
{
public:
    virtual ~Database(){}
    virtual int const* getNbTrainingExemple() const=0;
    virtual int const* getNbValidationExemple() const=0;
    virtual int const* getNbTestExemple() const=0;
    virtual int getInputSize() const=0;
    virtual int getOutputSize() const=0;
    virtual void loadTrainingInput(Eigen::MatrixXd &input,Eigen::MatrixXd &sortieAttendue,int debut,int nombre) const=0;
    virtual void loadValidationInput(Eigen::MatrixXd &input,Eigen::MatrixXd &sortieAttendue) const=0;
    virtual void loadTestInput(Eigen::MatrixXd &input,Eigen::MatrixXd &sortieAttendue) const=0;
private:

};

template <typename  T> //patron de fonction pour ne pas avoir à déclarer les types
class DatabaseT: public Database
{
public:
    DatabaseT(std::string dataAddress)
    {
        std::ifstream data(dataAddress, std::ios::in);

        if(data)
        {
            std::string type;
            data>>type;
            data>>m_inputSize>>m_outputSize>>m_nbTrainingExemple>>m_nbValidationExemple>>m_nbTestExemple;

            m_trainingData=new  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(m_inputSize,m_nbTrainingExemple);
            m_resultTrainingData=new  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(m_outputSize,m_nbTrainingExemple);

            m_validationData=new  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(m_inputSize,m_nbValidationExemple);
            m_resultValidationData=new  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(m_outputSize,m_nbValidationExemple);

            m_testData=new  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(m_inputSize,m_nbTestExemple);
            m_resultTestData=new  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(m_outputSize,m_nbTestExemple);

            for(int j{0};j<m_nbTrainingExemple;j++)
            {
                for(int i{0};i<m_inputSize;i++)
                {
                    data>>(*m_trainingData)(i,j);
                }

                for(int i{0};i<m_outputSize;i++)
                {
                    data>>(*m_resultTrainingData)(i,j);
                }
            }

            for(int j{0};j<m_nbValidationExemple;j++)
            {
                for(int i{0};i<m_inputSize;i++)
                {
                    data>>(*m_validationData)(i,j);
                }

                for(int i{0};i<m_outputSize;i++)
                {
                    data>>(*m_resultValidationData)(i,j);
                }
            }

            for(int j{0};j<m_nbTestExemple;j++)
            {
                for(int i{0};i<m_inputSize;i++)
                {
                    data>>(*m_testData)(i,j);
                }

                for(int i{0};i<m_outputSize;i++)
                {
                    data>>(*m_resultTestData)(i,j);
                }
            }
            data.close();
        }
        else
            std::cerr << "data pas ouvert" << std::endl;
    }

    virtual ~DatabaseT()
    {
        delete m_trainingData;
        delete m_resultTrainingData;
        delete m_validationData;
        delete m_resultValidationData;
        delete m_testData;
        delete m_resultTestData;
    }

    virtual int const* getNbTrainingExemple() const
    {
        return &m_nbTrainingExemple;
    }

    virtual int const* getNbValidationExemple() const
    {
        return &m_nbValidationExemple;
    }

    virtual int const* getNbTestExemple() const
    {
        return &m_nbTestExemple;
    }

    virtual int getInputSize() const
    {
        return m_inputSize;
    }

    virtual int getOutputSize() const
    {
        return m_outputSize;
    }


    virtual void loadTrainingInput(Eigen::MatrixXd &input,Eigen::MatrixXd &sortieAttendue,int debut,int nombre) const
    {
        std::srand(std::time(nullptr));
        for(int j{0};j<nombre;j++)
        {
            int i;
            if(debut+j<m_nbTrainingExemple)
                i=debut+j;
            else
                i=std::rand()%m_nbTrainingExemple;

            input.col(j)=m_trainingData->col(i).template cast<double>();
            sortieAttendue.col(j)=m_resultTrainingData->col(i).template cast<double>();
        }
    }

    virtual void loadValidationInput(Eigen::MatrixXd &input,Eigen::MatrixXd &sortieAttendue) const
    {
        input=m_validationData->template cast<double>();
        sortieAttendue=m_resultValidationData->template cast<double>();
    }
    virtual void loadTestInput(Eigen::MatrixXd &input,Eigen::MatrixXd &sortieAttendue) const
    {
        input=m_testData->template cast<double>();
        sortieAttendue=m_resultTestData->template cast<double>();
    }

private:
    int m_inputSize;
    int m_outputSize;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *m_trainingData{0};
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *m_resultTrainingData{0};
    int m_nbTrainingExemple;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *m_validationData{0};
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *m_resultValidationData{0};
    int m_nbValidationExemple;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *m_testData{0};
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> *m_resultTestData{0};
    int m_nbTestExemple;

};

#endif // DATABASE_H
