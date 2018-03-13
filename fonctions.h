#ifndef FONCTIONS_H
#define FONCTIONS_H

#include <cmath>
#include <Eigen/Core>

class ActFunction
{
public:
    virtual Eigen::MatrixXd const operator() (Eigen::MatrixXd const&mat)=0; //virtual c'est pour utiliser l'opération ailleurs
    virtual Eigen::MatrixXd const prime(Eigen::MatrixXd const&mat)=0;
    virtual ~ActFunction(){}
};

class Sigmoid: public ActFunction
{
public:
    virtual Eigen::MatrixXd const operator() (Eigen::MatrixXd const&mat);//peut ajouter petit +ax pour eviter des lieux plats
    virtual Eigen::MatrixXd const prime(Eigen::MatrixXd const&mat);
    virtual ~Sigmoid(){}
};

class SoftMax: public ActFunction // distribution de probabilité bien pour lesneuronnes de sortie pour un probleme de classification avec log_likelihood cost
{
public:
    virtual Eigen::MatrixXd const operator() (Eigen::MatrixXd const&mat);//peut ajouter petit +ax pour eviter des lieux plats
    virtual Eigen::MatrixXd const prime(Eigen::MatrixXd const&mat);
    virtual ~SoftMax(){}
};

class Tanh: public ActFunction
{
public:
    virtual Eigen::MatrixXd const operator() (Eigen::MatrixXd const&mat);//peut ajouter petit +ax pour eviter des lieux plats
    virtual Eigen::MatrixXd const prime(Eigen::MatrixXd const&mat);
    virtual ~Tanh(){}
};

class UpTanh: public ActFunction//conseillé sur un site
{
public:
    virtual Eigen::MatrixXd const operator() (Eigen::MatrixXd const&mat);//peut ajouter petit +ax pour eviter des lieux plats
    virtual Eigen::MatrixXd const prime(Eigen::MatrixXd const&mat);
    virtual ~UpTanh(){}
};

class CostFunction
{
public:
    virtual double operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput)=0;
    virtual Eigen::MatrixXd const gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput)=0;
    virtual ~CostFunction(){}
};
class CrossEntropy: public CostFunction//bien avec sigmoid
{
public:
    virtual double operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput);
    virtual Eigen::MatrixXd const gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput);
    virtual ~CrossEntropy(){}
};

class Quadratic: public CostFunction//basic
{
public:
    virtual double operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput);
    virtual Eigen::MatrixXd const gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput);
    virtual ~Quadratic(){}
};

class LogLikelihood: public CostFunction//bien avec softmax desiredOutput doit être que des pas bon(0 en principe)  et un bon (1)
{
public:
    virtual double operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput);
    virtual Eigen::MatrixXd const gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput);
    virtual ~LogLikelihood(){}
};

#endif // FONCTIONS_H
