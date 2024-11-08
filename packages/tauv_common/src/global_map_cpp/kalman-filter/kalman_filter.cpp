#include "./kalman_filter.hpp"
#include <limits>
#include <cmath>

using namespace std;

KalmanFilter::KalmanFilter(string det_tag, Eigen::VectorXd initial_estimate, double confidence)
{
    estimate = initial_estimate;
    tag = det_tag;
    dataDim = initial_estimate.size();
    max_confidence = confidence;

    inf = std::numeric_limits<double>::infinity();
    //used to represent huge variance in measurements
    inf_mat = Eigen::MatrixXd::Identity(dataDim, dataDim);
    for(size_t i=0; i<dataDim; i++){inf_mat(i,i)=inf;}

    readParams();
}

shared_ptr<KalmanFilter> KalmanFilter::copy(string new_tag)
{
    shared_ptr<KalmanFilter> NEW (new KalmanFilter(new_tag, estimate, max_confidence));
    NEW->Pk = Pk;
    NEW->Q = Q;
    NEW->R = R;
    NEW->A = A;
    NEW->B = B;
    NEW->H = H;
    return NEW;
}

void KalmanFilter::readParams()
{
    //read in the kalman filter ros parameters
    Pk = getParam("Pk");
    Q = getParam("Q");
    R = getParam("R");
    A = getParam("A");
    B = getParam("B");
    H = getParam("H");
}

Eigen::MatrixXd KalmanFilter::getParam(string property)
{
    vector<double> vec;

    //try tag-specific
    if(!ros::param::get("kalman_params/"+tag+"/"+property, vec)){
        //try default
        if(!ros::param::get("kalman_params/default/"+property, vec))
        {
            //initialize to identity
            return Eigen::MatrixXd::Identity(dataDim,dataDim);
        }
    }

    Eigen::Map<Eigen::MatrixXd> MAT(vec.data(), dataDim, dataDim);
    return MAT;
}

void KalmanFilter::makeMatrixUpdates(Eigen::VectorXd newEstimate, Eigen::VectorXd zk, Eigen::MatrixXd R_cur)
{
    Eigen::MatrixXd PkEstimate = (A*Pk * A.transpose()) + Q;

    Eigen::MatrixXd kGain = (PkEstimate*H.transpose())*((H*PkEstimate*H.transpose())+R_cur).inverse();

    estimate = newEstimate+kGain*(zk-(H*newEstimate));
    Pk = (Eigen::MatrixXd::Identity(dataDim, dataDim) - (kGain*H))*PkEstimate;
}

void KalmanFilter::makeUpdates(Eigen::VectorXd newEstimate, Eigen::VectorXd zk, size_t data_dim, double confidence)
{
    max_confidence = max(max_confidence, confidence);

    double cov_scale = 1.0/confidence;
    Eigen::MatrixXd R_cur = isnan(cov_scale) || isinf(cov_scale) ? inf_mat : (cov_scale)*R;
    R_cur.bottomRows(dataDim-data_dim)=inf_mat.bottomRows(dataDim-data_dim);
    
    makeMatrixUpdates(newEstimate, zk, R_cur);
}

void KalmanFilter::updateEstimate(Eigen::VectorXd zk, Eigen::VectorXd uk, size_t data_dim, double confidence)
{
    
    Eigen::VectorXd newEstimate = (A * estimate) + (B*uk);
    makeUpdates(newEstimate, zk, data_dim, confidence);
}

void KalmanFilter::updateEstimate(Eigen::VectorXd zk, size_t data_dim, double confidence)
{
    Eigen::VectorXd newEstimate = A * estimate; //independent of state B = 0
    makeUpdates(newEstimate, zk, data_dim, confidence);
}

void KalmanFilter::reset(Eigen::VectorXd zk, double confidence)
{
    Pk = getParam("Pk");
    estimate = zk;
    max_confidence = confidence;
}

Eigen::VectorXd KalmanFilter::getEstimate(){return estimate;}
double KalmanFilter::getConfidence(){return max_confidence;}