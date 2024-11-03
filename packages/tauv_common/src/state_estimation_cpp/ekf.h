#include <array>
#include <eigen3/Eigen/Dense>

#pragma once

class Ekf {
  public:
    Ekf();

    enum StateIndex { X, Y, Z, VX, VY, VZ, AX, AY, AZ, ROLL, PITCH, YAW, VROLL, VPITCH, VYAW };

    void set_dvl_offset(Eigen::Vector3d &dvl_offset) { this->dvl_offset = dvl_offset; };
    void set_process_covariance(Eigen::Matrix<double, 15, 1> &process_covariance) { this->process_covariance = Eigen::DiagonalMatrix<double, 15, 15>(process_covariance); };

    void get_state(Eigen::Matrix<double, 15, 1> &state);
    void set_state(Eigen::Matrix<double, 15, 1> &state);

    void get_cov(Eigen::Matrix<double, 15, 15> &cov);
    void set_cov(Eigen::Matrix<double, 15, 15> &cov);

    void set_time(double time);

    void set_reference_yaw(double yaw);

    void get_state_fields(double time,
        Eigen::Vector3d &position,
        Eigen::Vector3d &velocity,
        Eigen::Vector3d &acceleration,
        Eigen::Vector3d &orientation,
        Eigen::Vector3d &angular_velocity);

    void handle_imu_measurement(double time,
        const Eigen::Vector3d &orientation,
        const Eigen::Vector3d &rate_of_turn,
        const Eigen::Vector3d &linear_acceleration,
        const Eigen::Matrix<double, 9, 1> &covariance);
    void handle_dvl_measurement(double time, const Eigen::Vector3d &velocity, const Eigen::Vector3d &covariance);
    void handle_depth_measurement(double time, double depth, double covariance);

  private:
    Eigen::Vector3d dvl_offset;
    Eigen::Matrix<double, 15, 15> process_covariance;

    bool initialized;

    double time;

    Eigen::Matrix<double, 15, 1> state;
    Eigen::Matrix<double, 15, 15> cov;

    double reference_yaw;

    void predict(double time);
    void update(Eigen::VectorXi &fields, Eigen::VectorXd &inn, Eigen::VectorXd &cov);

    void extrapolate_state(double dt, Eigen::Matrix<double, 15, 1> &old_state, Eigen::Matrix<double, 15, 1> &new_state);
    void extrapolate_cov(double dt, Eigen::Matrix<double, 15, 15> &old_cov, Eigen::Matrix<double, 15, 15> &new_cov);

    void wrap_angles(Eigen::Vector3d &state);
    void wrap_angles(Eigen::VectorXi &fields, Eigen::Matrix<double, 15, 1> &state);
    void wrap_angles(Eigen::VectorXi &fields, Eigen::VectorXd &state);

    void get_dvl_tangential_velocity(Eigen::Matrix<double, 3, 1> &v);

    void get_H(Eigen::VectorXi &fields, Eigen::Matrix<double, Eigen::Dynamic, 15> &H);
    void get_F(double dt, Eigen::Matrix<double, 15, 15> &F);
    void get_J(double dt, Eigen::Matrix<double, 15, 15> &J);
    double get_partial(double xc, double yc, double zc, double dt);
};

Eigen::Quaterniond rpy_to_quat(const Eigen::Vector3d &rpy);