#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <vector>
#include <string>

/**
 * General linear Kalman Filter implementation created for filtering positions of static objects in 
 * the mapping system but no backwards tie to system exists so can be generalized.
 * All notation and formulas found in: http://www.cs.unc.edu/~tracker/media/pdf/SIGGRAPH2001_CoursePack_08.pdf
**/

using namespace std;

class KalmanFilter
{
    public:
        KalmanFilter(string tag, Eigen::VectorXd initial_estimate, double confidence=1.0);
        virtual ~KalmanFilter(){};
        shared_ptr<KalmanFilter> copy(string new_tag);

        virtual void updateEstimate(Eigen::VectorXd zk, Eigen::VectorXd uk, size_t data_dim, double confidence = 1.0); //state dependent
        virtual void updateEstimate(Eigen::VectorXd zk, size_t data_dim, double confidence = 1.0); //state independent

        Eigen::VectorXd getEstimate();
        double getConfidence();
        
        void reset(Eigen::VectorXd zk, double confidence = 1.0);
        
        Eigen::VectorXd estimate;
        Eigen::MatrixXd Pk;
        Eigen::MatrixXd Q;
        Eigen::MatrixXd R;
        Eigen::MatrixXd A;
        Eigen::MatrixXd B;
        Eigen::MatrixXd H;
        Eigen::MatrixXd inf_mat;

        size_t dataDim;

    private:
        void readParams();

        double inf;
        string tag;
        double max_confidence; //if max confidence is 0 will be completely overwritten with any non-zero estimate!
        Eigen::MatrixXd getParam(string property);
        void makeMatrixUpdates(Eigen::VectorXd newEstimate, Eigen::VectorXd zk, Eigen::MatrixXd R_cur);
        void makeUpdates(Eigen::VectorXd newEstimate, Eigen::VectorXd zk, size_t data_dim, double confidence);
};