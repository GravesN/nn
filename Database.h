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
    virtual Eigen::MatrixXd const getResultTestOutput() const=0;
    virtual Eigen::MatrixXd const getTestInput() const=0;

    virtual int getInputSize() const=0;
    virtual int getOutputSize() const=0;
    virtual void loadTrainingInput(Eigen::MatrixXd &input,Eigen::MatrixXd &sortieAttendue,int debut,int nombre) const=0;
    virtual void loadValidationInput(Eigen::MatrixXd &input,Eigen::MatrixXd &sortieAttendue) const=0;

    virtual std::string nom() const=0;
private:

};

template <typename  T> //patron de fonction pour ne pas avoir à déclarer les types
class DatabaseT: public Database
{
public:

    DatabaseT(std::string dataAddress)
    {
        /// construit la base de données ///
        m_nom=dataAddress;
        std::ifstream data(dataAddress, std::ios::in);

        if(data)
        {
            std::string type;
            std::string dossier;
            data>>type;
            data>>m_inputSize>>m_outputSize>>m_nbTrainingExemple>>m_nbValidationExemple>>m_nbTestExemple>>dossier;

            m_trainingData=new  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(m_inputSize,m_nbTrainingExemple);
            m_resultTrainingData=new  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(m_outputSize,m_nbTrainingExemple);

            m_validationData=new  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(m_inputSize,m_nbValidationExemple);
            m_resultValidationData=new  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(m_outputSize,m_nbValidationExemple);

            m_testData=new  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(m_inputSize,m_nbTestExemple);
            m_resultTestData=new  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(m_outputSize,m_nbTestExemple);

            std::ifstream train(dossier+"nomsTrain.txt",std::ios::in);
            std::string nomFichier;
            std::cout << m_inputSize << "  " << m_outputSize << std::endl;

            for(int j{0};j<m_nbTrainingExemple;j++)
            {
                train >> nomFichier;
                std::ifstream flux(dossier+"train/"+nomFichier,std::ios::in);
                for(int i{0};i<m_inputSize;i++)
                {
                    flux >> (*m_trainingData)(i,j);
                }
                for(int i{0};i<m_outputSize;i++)
                {
                    flux>>(*m_resultTrainingData)(i,j);
                }
                flux.close();
            }
            train.close();
            std::cout << "train ok" << std::endl;

            std::ifstream validation(dossier+"nomsValidation.txt",std::ios::in);
            for(int j{0};j<m_nbValidationExemple;j++)
            {
                validation >> nomFichier;
                //std::cout << nomFichier << std::endl;
                std::ifstream flux(dossier+"validation/"+nomFichier,std::ios::in);
                for(int i{0};i<m_inputSize;i++)
                {
                    flux>>(*m_validationData)(i,j);
                   // std:: cout << (*m_validationData)(i,j) << "  ";
                }
                for(int i{0};i<m_outputSize;i++)
                {
                    flux>>(*m_resultValidationData)(i,j);
                    //std::cout << (*m_resultValidationData)(i,j) << "  ";
                }
                flux.close();
            }
            //std::cout << std::endl << *m_validationData << std::endl << std::endl << *m_resultValidationData << std::endl;
            validation.close();
            std::cout << "validation ok" << std::endl;

            std::ifstream test(dossier+"nomsTest.txt",std::ios::in);
            for(int j{0};j<m_nbTestExemple;j++)
            {
                test >> nomFichier;
                std::ifstream flux(dossier+"validation/"+nomFichier,std::ios::in);
                for(int i{0};i<m_inputSize;i++)
                {
                    flux>>(*m_testData)(i,j);
                }

                for(int i{0};i<m_outputSize;i++)
                {
                    flux>>(*m_resultTestData)(i,j);
                }

                flux.close();
            }
            test.close();
            std::cout << "test ok" << std::endl;
            data.close();
        }
        else
            std::cerr << "data pas ouvert" << std::endl;
    }

    virtual ~DatabaseT()
    {
        /// destructeur de type ///
        delete m_trainingData;
        delete m_resultTrainingData;
        delete m_validationData;
        delete m_resultValidationData;
        delete m_testData;
        delete m_resultTestData;
    }

    virtual int const* getNbTrainingExemple() const
    {
        /// renvoie le nombre d'exemples pour l'entraînement ///
        return &m_nbTrainingExemple;
    }

    virtual int const* getNbValidationExemple() const
    {
        /// renvoie le nombre d'exemples pour la validation ///
        return &m_nbValidationExemple;
    }

    virtual int const* getNbTestExemple() const
    {
        /// renvoie le nombre d'exemples pour le test ///
        return &m_nbTestExemple;
    }

    virtual int getInputSize() const
    {
        /// renvoie la taille de l'entrée ///
        return m_inputSize;
    }

    virtual int getOutputSize() const
    {
        /// renvoie la taille de la sortie ///
        return m_outputSize;
    }

    virtual Eigen::MatrixXd const getResultTestOutput() const
    {
        /// renvoie la matrice de flottants contenant la sortie attandue pour le test (double précision) ///
        return (m_resultTestData->template cast<double>());
    }

    virtual Eigen::MatrixXd const getTestInput() const
    {
        /// renvoie la matrice de flottants contenant l'entrée attendue pour le test (double précision) ///
        return (m_testData->template cast<double>());
    }

    virtual void loadTrainingInput(Eigen::MatrixXd &input,Eigen::MatrixXd &sortieAttendue,int debut,int nombre) const
    {
        /// charge les données pour l'entraînement dans les matrices fournies, convertis en flottants (double précision) ///
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
        /// charge les données pour la validation dans les matrices fournies, convertis en flottants (double précision) ///
        input=m_validationData->template cast<double>();
        sortieAttendue=m_resultValidationData->template cast<double>();
    }

    virtual std::string nom() const
    {
        /// renvoie le nom de la base de données ///
        return m_nom;
    }

private:
    std::string m_nom;
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
