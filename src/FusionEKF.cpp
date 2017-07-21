#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  // measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0,      0.0225;

  // measurement covariance matrix - radar
  R_radar_ << 0.09, 0,      0,
              0,    0.0009, 0,
              0,    0,      0.09;

  // measurement matrix
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // set the acceleration noise components
  noise_ax = 9;
  noise_ay = 9;

  // initialize EKF matrices and vectors

  // create a 4D state vector, we don't know yet the values of the x state
  ekf_.x_ = VectorXd(4);

  ekf_.Q_ = MatrixXd(4, 4);

  // state covariance matrix P
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ <<  1, 0, 0,    0,
              0, 1, 0,    0,
              0, 0, 1000, 0,
              0, 0, 0,    1000;

  // the initial transition matrix F
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ <<  1, 0, 1, 0,
              0, 1, 0, 1,
              0, 0, 1, 0,
              0, 0, 0, 1;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    cout << "Kalman Filter Initialization " << endl;
    // first measurement
    // NOTE: covariance matrices are initialized in the constructor
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // convert radar data from polar to cartesian coordinates and initialize state with zero velocity
      float rho = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      ekf_.x_ << rho * cos(phi), rho * sin(phi), 0, 0;
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // set the state with the initial location and zero velocity
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }
    // set initial timestamp
    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;  // dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  // pre-compute a set of terms to avoid repeated calculation
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  dt_3 = dt_3 / 2;
  dt_4 = dt_4 / 4;
  float dt_3_ax = dt_3 * noise_ax;
  float dt_3_ay = dt_3 * noise_ay;

  // update the state transition matrix F according to the new elapsed time
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // update the process noise covariance matrix Q
  ekf_.Q_ <<  dt_4*noise_ax, 0,             dt_3_ax,       0,
              0,             dt_4*noise_ay, 0,             dt_3_ay,
              dt_3_ax,       0,             dt_2*noise_ax, 0,
              0,             dt_3_ay,       0,             dt_2*noise_ay;


  // predict
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar measurement update
    tools.CalculateJacobian(Hj_, ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser measurement update
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "\nx_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
