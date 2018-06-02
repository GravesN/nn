#include "fonctions.h"


std::string intToString(int n)
{
    std::ostringstream str;

    str << n;

    return str.str();
}

Eigen::MatrixXd const UpTanh::operator() (Eigen::MatrixXd const&mat) const
{
    return 1.7159*((2./3)*mat).array().tanh();
}

Eigen::MatrixXd const UpTanh::prime(Eigen::MatrixXd const&mat) const//prime c'est pour la dérivée
{
    return 1.7159*2/3.*((2/3.)*mat).array().cosh().square().inverse();
}

std::string UpTanh::nom() const
{
    return "UpTanh";
}

Eigen::MatrixXd const Tanh::operator() (Eigen::MatrixXd const&mat) const
{
    return mat.array().tanh();
}

Eigen::MatrixXd const Tanh::prime(Eigen::MatrixXd const&mat) const
{
    return mat.array().cosh().square().inverse();
}

std::string Tanh::nom() const
{
    return "Tanh";
}

Eigen::MatrixXd const Sigmoid::operator() (Eigen::MatrixXd const&mat) const
{
    return ((-mat.array()).exp()+1).inverse();
}

Eigen::MatrixXd const Sigmoid::prime(Eigen::MatrixXd const&mat) const
{
    return (mat.array().exp()+2+(-mat).array().exp()).inverse();
}

std::string Sigmoid::nom() const
{
    return "Sigmoid";
}

Eigen::MatrixXd const ReLU::operator() (Eigen::MatrixXd const&mat) const
{
    return mat.cwiseMax(0.01*mat);
}

Eigen::MatrixXd const ReLU::prime(Eigen::MatrixXd const&mat) const
{
    return mat.array().unaryExpr([](double x) { if(x>=0)return 1.;else return 0.01;});
}

std::string ReLU::nom() const
{
    return "ReLU";
}

Eigen::MatrixXd const SoftMax::operator() (Eigen::MatrixXd const&mat) const
{
    return mat.array().exp().rowwise()*mat.array().exp().colwise().sum().inverse();
}

Eigen::MatrixXd const SoftMax::prime(Eigen::MatrixXd const&mat) const
{
    SoftMax s;
    Eigen::MatrixXd const a = s(mat);
    //auto a=(this->operator()(mat)).array();
    return a - a.array().square().matrix();
}

std::string SoftMax::nom() const
{
    return "SoftMax";
}

double CrossEntropy::operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const
{
    Eigen::MatrixXd a(1,output.cols());
    for(int i{0};i<output.cols();i++)
            a(0,i)=1;
    return -1.0/(output.cols())*(output.array()*(desiredOutput.array().log())+(a-output).array()*(a-desiredOutput).array().log()).sum();
}

Eigen::MatrixXd const CrossEntropy::gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const
{
    Eigen::MatrixXd a(1,output.cols());
    for(int i{0};i<output.cols();i++)
            a(0,i)=1;
    return -1.0/(output.cols())*(output.array()*(desiredOutput.array().inverse()) - ((a-output).array()*(a-desiredOutput).array().inverse()));
}

std::string CrossEntropy::nom() const
{
    return "CrossEntropy";
}

double Quadratic::operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const
{
    double norme=(output-desiredOutput).norm();
    return (norme*norme)/output.cols()/2;
}

Eigen::MatrixXd const Quadratic::gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const
{
    return (output-desiredOutput)/output.cols();
}

std::string Quadratic::nom() const
{
    return "Quadratic";
}

double LogLikelihood::operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const
{
    return -1.0/(output.cols())*std::log(output.cwiseProduct(desiredOutput).colwise().maxCoeff().prod());
}

Eigen::MatrixXd const LogLikelihood::gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const
{

}

std::string LogLikelihood::nom() const
{
    return "LogLikelihood";
}
