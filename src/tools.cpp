#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  // accumulate squared residuals
  for (unsigned int i = 0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];

    // coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  // calculate the mean
  rmse = rmse / estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();

  // return the result
  return rmse;
}

void Tools::CalculateJacobian(MatrixXd& Hj, const VectorXd& x_state) {
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // pre-compute a set of terms to avoid repeated calculation, part 1
  float pxpy = px * px + py * py;

  // check division by zero
  if (fabs(pxpy) < 0.0001) {
    cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    cout << px << " " << py << endl;
    return;
  }

  // pre-compute a set of terms to avoid repeated calculation, part 2
  float pxpy_sqrt = sqrt(pxpy);
  float pxpy_sqrt_3 = pxpy * pxpy_sqrt;
  float vxpy = vx * py;
  float vypx = vy * px;

  float px_pxpy_sqrt = px / pxpy_sqrt;
  float py_pxpy_sqrt = py / pxpy_sqrt;

  // compute the Jacobian matrix
  Hj << px_pxpy_sqrt,                     py_pxpy_sqrt,                     0,            0,
        -py / pxpy,                       px / pxpy,                        0,            0,
        py * (vxpy - vypx) / pxpy_sqrt_3, px * (vypx - vxpy) / pxpy_sqrt_3, px_pxpy_sqrt, py_pxpy_sqrt;
}
