#ifndef DATABASE_H
#define DATABASE_H

#include <Eigen/Dense>
#include <string>
#include <iostream>

class Database
{
public:
    Database(std::string trucPourData);
    virtual ~Database();

private:
    Eigen::MatrixXd m_trainingData;
    Eigen::MatrixXd m_resultTrainingData;
    Eigen::MatrixXd m_validationData;
    Eigen::MatrixXd m_resultValidationData;
    Eigen::MatrixXd m_testData;
    Eigen::MatrixXd m_resultTestData;

};

#endif // DATABASE_H
