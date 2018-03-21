#include "fonctions.h"

Eigen::MatrixXd const UpTanh::operator() (Eigen::MatrixXd const&mat)
{
    return 1.7159*((2./3)*mat).array().tanh();
}

Eigen::MatrixXd const UpTanh::prime(Eigen::MatrixXd const&mat)//prime c'est pour la dérivée
{
    return 1.7159*2/3.*((2/3.)*mat).array().cosh().pow(2).inverse();
}

Eigen::MatrixXd const Tanh::operator() (Eigen::MatrixXd const&mat)
{
    return mat.array().tanh();
}

Eigen::MatrixXd const Tanh::prime(Eigen::MatrixXd const&mat)
{
    return mat.array().cosh().pow(2).inverse();
}

Eigen::MatrixXd const Sigmoid::operator() (Eigen::MatrixXd const&mat)
{
    return ((-mat.array()).exp()+1).inverse();
}

Eigen::MatrixXd const Sigmoid::prime(Eigen::MatrixXd const&mat)
{
    return (mat.array().exp()+2+(-mat).array().exp()).inverse();
}

Eigen::MatrixXd const SoftMax::operator() (Eigen::MatrixXd const&mat)
{
    return mat.array().exp().rowwise()*mat.array().exp().colwise().sum().inverse();
}

Eigen::MatrixXd const SoftMax::prime(Eigen::MatrixXd const&mat)
{
    auto a=(this->operator()(mat)).array();
    return a-a.pow(2);
}

double CrossEntropy::operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput)
{
    return -1/(output.cols())*((desiredOutput.array()*Eigen::log(output.array())+(1-desiredOutput.array())*Eigen::log(1-output.array())).sum())
}

Eigen::MatrixXd const CrossEntropy::gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput)
{
    return -1/(output.cols())*((desiredOutput.array()/output.array() - (1-desiredOutput.array())/(1-output.array()))).rowwise().sum()
}

double Quadratic::operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput)
{

}

Eigen::MatrixXd const Quadratic::gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput)
{
    return (output-desiredOutput).rowwise().sum()/output.rows();
}

double LogLikelihood::operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput)
{
    return -log(output.cwiseProduct(desiredOutput).colwise().maxCoeff().prod());
}

Eigen::MatrixXd const LogLikelihood::gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput)
{

}
