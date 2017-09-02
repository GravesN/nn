#ifndef FONCTIONS_H
#define FONCTIONS_H

#include <cmath>
#include <Eigen/Core>

class ActFunction
{
    public:
        virtual Eigen::MatrixXd operator() (Eigen::MatrixXd const&mat)=0;
        virtual Eigen::MatrixXd prime(Eigen::MatrixXd const&mat)=0;
};

class Sigmoid: public ActFunction
{
    public:
        virtual Eigen::MatrixXd operator() (Eigen::MatrixXd const&mat);//peut ajouter petit +ax pour eviter des lieux plats
        virtual Eigen::MatrixXd prime(Eigen::MatrixXd const&mat);
};

class SoftMax: public ActFunction // distribution de probabilité bien pour lesneuronnes de sortie pour un probleme de classification avec log_likelihood cost
{
    public:
        virtual Eigen::MatrixXd operator() (Eigen::MatrixXd const&mat);//peut ajouter petit +ax pour eviter des lieux plats
        virtual Eigen::MatrixXd prime(Eigen::MatrixXd const&mat);
};

class Tanh: public ActFunction
{
    public:
        virtual Eigen::MatrixXd operator() (Eigen::MatrixXd const&mat);//peut ajouter petit +ax pour eviter des lieux plats
        virtual Eigen::MatrixXd prime(Eigen::MatrixXd const&mat);
};

class UpTanh: public ActFunction//conseillé sur un site
{
    public:
        virtual Eigen::MatrixXd operator() (Eigen::MatrixXd const&mat);//peut ajouter petit +ax pour eviter des lieux plats
        virtual Eigen::MatrixXd prime(Eigen::MatrixXd const&mat);
};

class CostFunction
{
    public:
        virtual double operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput)=0;
        virtual Eigen::MatrixXd gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput)=0;
};
class CrossEntropy: public CostFunction//bien avec sigmoid
{
    public:
        virtual double operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput);
        virtual Eigen::MatrixXd gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput);
};

class Quadratic: public CostFunction//basic
{
    public:
        virtual double operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput);
        virtual Eigen::MatrixXd gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput);
};

class LogLikelihood: public CostFunction//bien avec softmax desiredOutput doit etre que des pas bon(0 en principe)  et un bon (1)
{
    public:
        virtual double operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput);
        virtual Eigen::MatrixXd gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput);
};
#endif // FONCTIONS_H
