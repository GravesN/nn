#ifndef FONCTIONS_H
#define FONCTIONS_H

#include <cmath>
#include <Eigen/Core>
#include <string>
#include <sstream>
#include <iostream>

std::string intToString(int n);

class ActFunction
{
public:
    virtual Eigen::MatrixXd const operator() (Eigen::MatrixXd const&mat) const =0 ; //virtual c'est pour utiliser l'opération ailleurs
    virtual Eigen::MatrixXd const prime(Eigen::MatrixXd const&mat) const =0 ;
    virtual ~ActFunction(){}
    virtual std::string nom()const=0;
};

class Sigmoid: public ActFunction
{
public:
    virtual Eigen::MatrixXd const operator() (Eigen::MatrixXd const&mat) const;//peut ajouter petit +ax pour eviter des lieux plats
    virtual Eigen::MatrixXd const prime(Eigen::MatrixXd const&mat) const;
    virtual ~Sigmoid(){}
    virtual std::string nom() const;
};

class SoftMax: public ActFunction // distribution de probabilité bien pour les neuronnes de sortie pour un probleme de classification avec log_likelihood cost
{
public:
    virtual Eigen::MatrixXd const operator() (Eigen::MatrixXd const&mat) const;//peut ajouter petit +ax pour eviter des lieux plats
    virtual Eigen::MatrixXd const prime(Eigen::MatrixXd const&mat) const;
    virtual ~SoftMax(){}
    virtual std::string nom() const;
};

class Tanh: public ActFunction
{
public:
    virtual Eigen::MatrixXd const operator() (Eigen::MatrixXd const&mat) const;//peut ajouter petit +ax pour eviter des lieux plats
    virtual Eigen::MatrixXd const prime(Eigen::MatrixXd const&mat) const;
    virtual ~Tanh(){}
    virtual std::string nom()const;

};

class UpTanh: public ActFunction//conseillé sur un site
{
public:
    virtual Eigen::MatrixXd const operator() (Eigen::MatrixXd const&mat) const;//peut ajouter petit +ax pour eviter des lieux plats
    virtual Eigen::MatrixXd const prime(Eigen::MatrixXd const&mat) const;
    virtual ~UpTanh(){}
    virtual std::string nom()const;
};

class ReLU: public ActFunction
{
public:
    virtual Eigen::MatrixXd const operator() (Eigen::MatrixXd const&mat) const;
    virtual Eigen::MatrixXd const prime(Eigen::MatrixXd const&mat) const;
    virtual ~ReLU(){}
    virtual std::string nom()const;
};

class CostFunction
{
public:
    virtual double operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const =0 ;
    virtual Eigen::MatrixXd const gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const =0 ;
    virtual ~CostFunction(){}
    virtual std::string nom()const=0;
};
class CrossEntropy: public CostFunction//bien avec sigmoid
{
public:
    virtual double operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const;
    virtual Eigen::MatrixXd const gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const;
    virtual ~CrossEntropy(){}
    virtual std::string nom()const;
};

class Quadratic: public CostFunction//basic
{
public:
    virtual double operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const;
    virtual Eigen::MatrixXd const gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const;
    virtual ~Quadratic(){}
    virtual std::string nom()const;
};

class LogLikelihood: public CostFunction//bien avec softmax desiredOutput doit être que des pas bon(0 en principe)  et un bon (1)
{
public:
    virtual double operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const;
    virtual Eigen::MatrixXd const gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const;
    virtual ~LogLikelihood(){}
    virtual std::string nom()const;
};

#endif // FONCTIONS_H
