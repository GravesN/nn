#include "fonctions.h"

Eigen::MatrixXd UpTanh::operator() (Eigen::MatrixXd const&mat)
{
    return 1.7159*((2./3)*mat).array().tanh();
}

Eigen::MatrixXd UpTanh::prime(Eigen::MatrixXd const&mat)
{
    return 1.7159*2/3.*((2/3.)*mat).array().cosh().pow(2).inverse();
}

Eigen::MatrixXd Tanh::operator() (Eigen::MatrixXd const&mat)
{
    return mat.array().tanh();
}

Eigen::MatrixXd Tanh::prime(Eigen::MatrixXd const&mat)
{
    return mat.array().cosh().pow(2).inverse();
}

Eigen::MatrixXd Sigmoid::operator() (Eigen::MatrixXd const&mat)
{
    return ((-mat.array()).exp()+1).inverse();
}

Eigen::MatrixXd Sigmoid::prime(Eigen::MatrixXd const&mat)
{
    return (mat.array().exp()+2+(-mat).array().exp()).inverse();
}

Eigen::MatrixXd SoftMax::operator() (Eigen::MatrixXd const&mat)
{
    return mat.array().exp().rowwise()*mat.array().exp().colwise().sum().inverse();
}

Eigen::MatrixXd SoftMax::prime(Eigen::MatrixXd const&mat)
{
    auto a=(this->operator()(mat)).array();
    return a-a.pow(2);
}

double CrossEntropy::operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput)
{

}

Eigen::MatrixXd CrossEntropy::gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput)
{

}

double Quadratic::operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput)
{
    return 0.5/output.cols()*pow((desiredOutput-output).norm(),2);
}

Eigen::MatrixXd Quadratic::gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput)
{
    return (output-desiredOutput).rowwise().sum()/output.rows();
}

double LogLikelihood::operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput)
{
    return -log(output.cwiseProduct(desiredOutput).colwise().maxCoeff().prod());
}

Eigen::MatrixXd LogLikelihood::gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput)
{

}
