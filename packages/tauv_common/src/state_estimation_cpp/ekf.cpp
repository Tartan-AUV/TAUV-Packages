#include "ekf.h"
#include <ros/ros.h>
#include <eigen3/Eigen/Dense>

Ekf::Ekf()
{
  this->cov = 1e-9 * Eigen::Matrix<double, 15, 15>::Identity();
  this->state = Eigen::Matrix<double, 15, 1>::Zero();
  this->reference_yaw = 0;
}

void Ekf::get_state(Eigen::Matrix<double, 15, 1> &state)
{
  state = this->state;
}

void Ekf::set_state(Eigen::Matrix<double, 15, 1> &state)
{
  this->state = state;
}

void Ekf::get_cov(Eigen::Matrix<double, 15, 15> &cov)
{
  cov = this->cov;
}

void Ekf::set_cov(Eigen::Matrix<double, 15, 15> &cov)
{
  this->cov = cov;
}

void Ekf::set_time(double time)
{
  this->time = time;
}

void Ekf::get_state_fields(double time, Eigen::Vector3d &position, Eigen::Vector3d &velocity, Eigen::Vector3d &acceleration, Eigen::Vector3d &orientation, Eigen::Vector3d &angular_velocity)
{
  using S = StateIndex;

  double dt = time - this->time;

  Eigen::Matrix<double, 15, 1> state = Eigen::Matrix<double, 15, 1>::Zero();
  this->extrapolate_state(dt, this->state, state);

  Eigen::Vector3d corrected_orientation { state(S::ROLL), state(S::PITCH), state(S::YAW) - this->reference_yaw };
  Eigen::Vector3d corrected_position {  cos(-this->reference_yaw) * state(S::X) - sin(-this->reference_yaw) * state(S::Y), sin(-this->reference_yaw) * state(S::X) + cos(-this->reference_yaw) * state(S::Y), state(S::Z) };
  this->wrap_angles(corrected_orientation);

  position << corrected_position.x(), corrected_position.y(), corrected_position.z();
  velocity << state(S::VX), state(S::VY), state(S::VZ);
  acceleration << state(S::AX), state(S::AY), state(S::AZ);
  orientation << corrected_orientation.x(), corrected_orientation.y(), corrected_orientation.z();
  angular_velocity << state(S::VROLL), state(S::VPITCH), state(S::VYAW);
}

void Ekf::handle_imu_measurement(double time,
    const Eigen::Vector3d &orientation,
    const Eigen::Vector3d &rate_of_turn,
    const Eigen::Vector3d &linear_acceleration,
    const Eigen::Matrix<double, 9, 1> &covariance)
{
  if (!this->initialized) {
    this->time = time;
    this->initialized = true;
  }

  this->predict(time);

  Eigen::VectorXi fields(9);
  fields << StateIndex::ROLL, StateIndex::PITCH, StateIndex::YAW, StateIndex::VROLL, StateIndex::VPITCH, StateIndex::VYAW, StateIndex::AX, StateIndex::AY, StateIndex::AZ;

  Eigen::Matrix<double, Eigen::Dynamic, 15> H = Eigen::Matrix<double, Eigen::Dynamic, 15>::Zero(9, 15);
  this->get_H(fields, H);

  // Need to remove the gravity component from linear_acceleration

  Eigen::VectorXd y(9);
  y << orientation, rate_of_turn, linear_acceleration;
  y -= H * this->state;

  Eigen::VectorXi angle_fields(3);
  angle_fields << 0, 1, 2;
//  this->wrap_angles(angle_fields, y);

  Eigen::VectorXd cov(9);
  cov << covariance;

  this->update(fields, y, cov);
}

void Ekf::handle_dvl_measurement(double time, const Eigen::Vector3d &velocity, const Eigen::Vector3d &covariance)
{
  if (!this->initialized) {
    this->time = time;
    this->initialized = true;
  }

  this->predict(time);

  Eigen::VectorXi fields(3);
  fields << StateIndex::VX, StateIndex::VY, StateIndex::VZ;

  Eigen::Matrix<double, Eigen::Dynamic, 15> H = Eigen::Matrix<double, Eigen::Dynamic, 15>::Zero(3, 15);
  this->get_H(fields, H);

  Eigen::Vector3d tangential_velocity = Eigen::Vector3d::Zero();
  this->get_dvl_tangential_velocity(tangential_velocity);

  Eigen::VectorXd y(3);
  y << velocity;
  y -= tangential_velocity;
  y -= H * this->state;

  Eigen::VectorXd cov(3);
  cov << covariance;

  this->update(fields, y, cov);
}

void Ekf::handle_depth_measurement(double time, double depth, double covariance)
{
  if (!this->initialized) {
    this->time = time;
    this->initialized = true;
  }

  this->predict(time);

  Eigen::VectorXi fields(1);
  fields << StateIndex::Z;

  Eigen::Matrix<double, Eigen::Dynamic, 15> H = Eigen::Matrix<double, Eigen::Dynamic, 15>::Zero(1, 15);
  this->get_H(fields, H);

  Eigen::VectorXd y(1);
  y << depth;
  y -= H * this->state;

  Eigen::VectorXd cov(1);
  cov << covariance;

  this->update(fields, y, cov);
}

void Ekf::predict(double time)
{
  double dt = time - this->time;
  this->time = time;

  Eigen::Matrix<double, 15, 1> new_state = Eigen::Matrix<double, 15, 1>::Zero();
  this->extrapolate_state(dt, this->state, new_state);
  this->state = new_state;

  Eigen::Matrix<double, 15, 15> new_cov = Eigen::Matrix<double, 15, 15>::Zero();
  this->extrapolate_cov(dt, this->cov, new_cov);
  this->cov = new_cov;
} 

void Ekf::update(Eigen::VectorXi &fields, Eigen::VectorXd &inn, Eigen::VectorXd &cov)
{
  Eigen::Matrix<double, Eigen::Dynamic, 15> H = Eigen::Matrix<double, Eigen::Dynamic, 15>::Zero(fields.rows(), 15);
  this->get_H(fields, H);

  Eigen::MatrixXd R = Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic>(cov);

  Eigen::MatrixXd S = (H * this->cov) * H.transpose() + R;
  Eigen::MatrixXd K = (this->cov * H.transpose()) * S.inverse();

  Eigen::Matrix<double, 15, 15> I = Eigen::Matrix<double, 15, 15>::Identity();

  this->state += K * inn;

  Eigen::VectorXi angle_fields(3); 
  angle_fields << StateIndex::ROLL, StateIndex::PITCH, StateIndex::YAW;
//  this->wrap_angles(angle_fields, this->state);

  this->cov = (I - K * H) * this->cov * (I - K * H).transpose();
  this->cov += (K * R) * K.transpose();
  this->cov = this->cov.cwiseAbs().cwiseMax(1e-9 * I);
}

void Ekf::extrapolate_state(double dt, Eigen::Matrix<double, 15, 1> &old_state, Eigen::Matrix<double, 15, 1> &new_state)
{
  using S = StateIndex;

  Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Zero();
  this->get_F(dt, F);

  new_state = F * old_state;

  Eigen::VectorXi fields(3); 
  fields << S::ROLL, S::PITCH, S::YAW;
//  this->wrap_angles(fields, new_state);
}

void Ekf::extrapolate_cov(double dt, Eigen::Matrix<double, 15, 15> &old_cov, Eigen::Matrix<double, 15, 15> &new_cov)
{
  Eigen::Matrix<double, 15, 15> J = Eigen::Matrix<double, 15, 15>::Zero();
  this->get_J(dt, J);

  new_cov = J * (old_cov * old_cov.transpose()) + (dt * this->process_covariance);
}

double wrap(double angle)
{
  // https://stackoverflow.com/a/29871193
  return -M_PI + fmod(2 * M_PI + fmod(angle + M_PI, 2 * M_PI), 2 * M_PI);
}

void Ekf::wrap_angles(Eigen::Vector3d &state)
{
  for (int i = 0; i < 3; i++) {
    state(i) = wrap(state(i));
  }
}

void Ekf::wrap_angles(Eigen::VectorXi &fields, Eigen::Matrix<double, 15, 1> &state)
{
  for (int i = 0; i < fields.rows(); i++) {
    state(fields(i)) = wrap(state(fields(i)));
  }
}

void Ekf::wrap_angles(Eigen::VectorXi &fields, Eigen::VectorXd &state)
{
  for (int i = 0; i < fields.rows(); i++) {
    state(fields(i)) = wrap(state(fields(i)));
  }
}

void Ekf::get_dvl_tangential_velocity(Eigen::Matrix<double, 3, 1> &v)
{
  using S = StateIndex;

  double cp = cos(this->state(S::PITCH));
  double sp = sin(this->state(S::PITCH));
  double cr = cos(this->state(S::ROLL));
  double sr = sin(this->state(S::ROLL));

  double vyaw = this->state(S::VYAW);
  double vpitch = this->state(S::VPITCH);
  double vroll = this->state(S::VROLL);

  Eigen::Vector3d w;
  w << (-sp * vyaw + vroll),
    (cp * sr * vyaw + cr * vpitch),
    (cp * cr * vyaw - sr * vpitch);

  v = w.cross(this->dvl_offset);
}

void Ekf::get_H(Eigen::VectorXi &fields, Eigen::Matrix<double, Eigen::Dynamic, 15> &H)
{
  H = Eigen::Matrix<double, Eigen::Dynamic, 15>(fields.rows(), 15) = Eigen::Matrix<double, Eigen::Dynamic, 15>::Zero(fields.rows(), 15); 

  for (int i = 0; i < fields.rows(); i++) {
    H(i, fields(i, 0)) = 1.0; 
  }
}

void Ekf::get_F(double dt, Eigen::Matrix<double, 15, 15> &F)
{
  using S = StateIndex;
  double cy = cos(this->state(S::YAW));
  double sy = sin(this->state(S::YAW));
  double cp = cos(this->state(S::PITCH));
  double sp = sin(this->state(S::PITCH));
  double tp = tan(this->state(S::PITCH));
  double cr = cos(this->state(S::ROLL));
  double sr = sin(this->state(S::ROLL));

  // State evolution matrix comes from robot_localization math
  // https://github.com/cra-ros-pkg/robot_localization/blob/ea976f9e65eac505f5cc7d3197ae080de4e5c10b/src/ekf.cpp#L259C5-L259C5

  // This propagates the last states forward in time, tracking 
  // Position in the world frame
  // Velocity in the body frame
  // Acceleration in the body frame
  // Orientation (euler angles) in the world frame
  // Angular velocity (axis-angle) in the body frame

  // See the GitHub wiki entry on kinematics math for more info on this

  F = Eigen::Matrix<double, 15, 15>::Identity();
  F(S::X, S::VX) = (cp * cy) * dt;
  F(S::X, S::VY) = (cy * sp * sr - cr * sy) * dt;
  F(S::X, S::VZ) = (cr * cy * sp + sr * sy) * dt;
  F(S::X, S::AX) = 0.5 * F(S::X, S::VX) * dt;
  F(S::X, S::AY) = 0.5 * F(S::X, S::VY) * dt;
  F(S::X, S::AZ) = 0.5 * F(S::X, S::VZ) * dt;
  F(S::Y, S::VX) = (cp * sy) * dt;
  F(S::Y, S::VY) = (cp * cy + sp * sr * sy) * dt;
  F(S::Y, S::VZ) = (-cy * sr + cr * sp * sy) * dt;
  F(S::Y, S::AX) = 0.5 * F(S::Y, S::VX) * dt;
  F(S::Y, S::AY) = 0.5 * F(S::Y, S::VY) * dt;
  F(S::Y, S::AZ) = 0.5 * F(S::Y, S::VZ) * dt;
  F(S::Z, S::VX) = (-sp) * dt;
  F(S::Z, S::VY) = (cp * sr) * dt;
  F(S::Z, S::VZ) = (cp * cr) * dt;
  F(S::Z, S::AX) = 0.5 * F(S::Z, S::VX) * dt;
  F(S::Z, S::AY) = 0.5 * F(S::Z, S::VY) * dt;
  F(S::Z, S::AZ) = 0.5 * F(S::Z, S::VZ) * dt;
  F(S::VX, S::AX) = dt;
  F(S::VY, S::AY) = dt;
  F(S::VZ, S::AZ) = dt; 
  F(S::YAW, S::VYAW) = (cr / cp) * dt;
  F(S::YAW, S::VPITCH) = (sr / cp) * dt;
  F(S::PITCH, S::VYAW) = (-sr) * dt;
  F(S::PITCH, S::VPITCH) = (cr) * dt;
  F(S::ROLL, S::VYAW) = (cr * tp) * dt;
  F(S::ROLL, S::VPITCH) = (sr * tp) * dt;
  F(S::ROLL, S::VROLL) = dt;
}

void Ekf::get_J(double dt, Eigen::Matrix<double, 15, 15> &J)
{
  using S = StateIndex;

  this->get_F(dt, J);

  double cy = cos(this->state(S::YAW));
  double sy = sin(this->state(S::YAW));
  double cp = cos(this->state(S::PITCH));
  double sp = sin(this->state(S::PITCH));
  double tp = tan(this->state(S::PITCH));
  double cr = cos(this->state(S::ROLL));
  double sr = sin(this->state(S::ROLL));

  J(S::X, S::YAW) = this->get_partial(-cp * sy,
      -cy * cr - sp * sy * sr,
      cy * sr - sp * sy * cr,
      dt);
    J(S::X, S::PITCH) = this->get_partial(-sp * cy,
        cp * cy * sr,
        cp * cy * cr,
        dt);
    J(S::X, S::ROLL) = this->get_partial(0,
        sp * cy * cr + sy * sr,
        sy * cr - sp * cy * sr,
        dt);
    J(S::Y, S::YAW) = this->get_partial(cp * cy,
        sp * cy * sr - sy * cr,
        sp * cy * sr + sy * sr,
        dt);
    J(S::Y, S::PITCH) = this->get_partial(-sp * sy,
        cp * sy * sr,
        cp * sy * cr,
        dt);
    J(S::Y, S::ROLL) = this->get_partial(0,
        sp * sy * cr - cy * sr,
        -cy * cr - sp * sy * sr,
        dt);
    J(S::Z, S::PITCH) = this->get_partial(-cp,
        -sp * sr,
        -sp * cr,
        dt);
    J(S::Z, S::ROLL) = this->get_partial(0,
        cp * sr,
        -cp * sr,
        dt);

  double vyaw = this->state(S::VYAW);
  double vpitch = this->state(S::VPITCH);

  J(S::YAW, S::PITCH) = (tp * sr / cp) * vpitch * dt + (tp * cr / cp) * vyaw * dt;
  J(S::YAW, S::ROLL) = (cr / cp) * vpitch * dt + (-sr / cp) * vyaw * dt;
  J(S::PITCH, S::ROLL) = (-sr) * vpitch * dt + (-cr) * vyaw * dt;
  J(S::ROLL, S::PITCH) = (sr / (cp * cp)) * vpitch * dt + (cr / (cp * cp)) * vyaw * dt;
  J(S::ROLL, S::ROLL) = 1 + (tp * cr) * vpitch * dt + (-tp * sr) * vyaw * dt;
}

double Ekf::get_partial(double xc, double yc, double zc, double dt)
{
  using S = StateIndex;

  double vx = this->state(S::VX);
  double vy = this->state(S::VY);
  double vz = this->state(S::VZ);
  double ax = this->state(S::AX);
  double ay = this->state(S::AY);
  double az = this->state(S::AZ);

  return (xc * vx + yc * vy + zc * vz) * dt + (xc * ax + yc * ay + zc * az) * 0.5 * dt * dt;
}

void Ekf::set_reference_yaw(double reference_yaw)
{
    this->reference_yaw = reference_yaw;
}

Eigen::Quaterniond rpy_to_quat(const Eigen::Vector3d &rpy)
{
    Eigen::AngleAxisd roll_angle(-rpy.x(), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch_angle(-rpy.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw_angle(-rpy.z(), Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = roll_angle * pitch_angle * yaw_angle;

    return q;
}