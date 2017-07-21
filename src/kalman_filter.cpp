#include "kalman_filter.h"
#include <math.h>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  // prediction step (same for regular and extended KF)
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::GeneralUpdate(const VectorXd &y) {
  // update steps after error calculation: y = z - z_pred
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  // execute the rest of the update set (same as with extended KF)
  GeneralUpdate(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  float pxpy = px * px + py * py;

  // check division by zero
  if (fabs(pxpy) < 0.0001) {
    std::cout << "UpdateEKF () - Error - Division by Zero" << std::endl;
    std::cout << px << " " << py << std::endl;
    return;
  }

  // non-linear transformation from cartesian to polar, part 1
  float rho = sqrt(pxpy);        // range rho
  float phi = atan2(py, px);              // angle phi
  float rho_dot = (px * vx + py * vy) / rho;  // range rate rho_dot

  // create prediction vector and assign polar coordinates
  VectorXd z_pred(3);
  z_pred << rho, phi, rho_dot;

  // calculate the error term of measurement and prediction
  VectorXd y = z - z_pred;
  // make adjustment to keep the error term for phi between -pi and pi
  float pi2 = 2 * M_PI;
  if (y(1) >= M_PI) {
    y(1) -= pi2;
  } else if (y(1) <= -M_PI) {
    y(1) += pi2;
  }
  // execute the rest of the update set (same as with regular KF)
  GeneralUpdate(y);
}
