#ifndef DATABASE_H
#define DATABASE_H

#include <Eigen/Dense>
#include <string>

class Database
{
    public:
        Database(std::string trucPourData);
        virtual ~Database();

    private:
        Eigen::matrixXd m_trainingData;
        Eigen::matrixXd m_resultTrainingData;
        Eigen::matrixXd m_validationData;
        Eigen::matrixXd m_resultValidationData;
        Eigen::matrixXd m_testData;
        Eigen::matrixXd m_resultTestData;

};

#endif // DATABASE_H
