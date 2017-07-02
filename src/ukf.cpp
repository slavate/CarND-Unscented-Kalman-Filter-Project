#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored
  use_laser_ = true;

  // if this is false, radar measurements will be ignored
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  //set state dimension
  n_x_ = 5;

  //set augmented dimension
  n_aug_ = 7;

  //define spreading parameter
  lambda_ = 3 - n_aug_;

  //set vector for weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0.0);
  //calculate weights
  double w1 = lambda_ / (lambda_ + n_aug_);
  double w2 = 0.5 / (lambda_ + n_aug_);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    if (i == 0) {
      weights_(i) = w1;
    }
    else {
      weights_(i) = w2;
    }
  }

  //cout << "weights_:" << endl << weights_ << endl;

  //Initialize sigma points matrix
  Xsig_ = MatrixXd(n_x_, 2 * n_x_ + 1);
  Xsig_.fill(0.0);

  //Initialize augmented sigma points matrix
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug_.fill(0.0);

  //Initialize predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0.0);

  NIS_radar_ = 0.0;
  NIS_laser_ = 0.0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  
  /*****************************************************************************
  *  Initialization
  ****************************************************************************/
  if (!is_initialized_) {

    // first measurement
    cout << "UKF: " << endl;
    
    // initialize state vector [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    x_ << 1.0, 1.0, 4.0, 0.0, 0.0;
    
    // initialize covariance matrix, we are uncertain about vel_abs, yaw_angle and yaw_rate
    P_ << 1.0, 0, 0, 0, 0,
          0, 1.0, 0, 0, 0,
          0, 0, std_a_*std_a_, 0, 0,
          0, 0, 0, std_yawdd_*std_yawdd_, 0,
          0, 0, 0, 0, 0.0;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float v = meas_package.raw_measurements_[2];
      
      x_(0) = rho*cos(phi);
      x_(1) = rho*sin(phi);
      x_(2) = v;
      P_(0, 0) = std_radr_ * std_radr_ + std_radphi_ * std_radphi_;
      P_(1, 1) = P_(0, 0);
      P_(2, 2) = std_radrd_*std_radrd_;
    }
    else {
      /**
      Initialize state. Set the state with the initial location and zero velocity
      */
      //cout << "Laser: " << measurement_pack.raw_measurements_;
      x_(0) = meas_package.raw_measurements_[0];
      x_(1) = meas_package.raw_measurements_[1];
      P_(0, 0) = std_laspx_ * std_laspx_;
      P_(1, 1) = std_laspy_ * std_laspy_;
    }
    
    time_us_ = meas_package.timestamp_;

    cout << "init state:" << endl << x_ << endl << endl;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  
  /*****************************************************************************
  *  Prediction
  ****************************************************************************/
  //compute the time elapsed between the current and previous measurements
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  /*****************************************************************************
  *  Update
  ****************************************************************************/
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    UpdateRadar(meas_package);
  }
  else {
    // Laser updates
    UpdateLidar(meas_package);
  }

  cout << "x_:" << endl << x_ << endl << endl;
  cout << "P_:" << endl << P_ << endl;
  cout << "#####################################################################" << endl << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  // augment sigma points
  AugmentedSigmaPoints();

  // predict sigma points
  SigmaPointPrediction(delta_t);

  //predict state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ = x_ + weights_(i)*Xsig_pred_.col(i);
  }

  //predict state covariance matrix
  P_.fill(0.0);
  //iterate over sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }

  //angle normalization
  while (x_(3)> M_PI) x_(3) -= 2.*M_PI;
  while (x_(3)<-M_PI) x_(3) += 2.*M_PI;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  // number of measured values, i.e. px, py
  int n_z = 2;

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);

    // measurement model
    Zsig(0, i) = px;         //laser px
    Zsig(1, i) = py;         //laser py
  }

  //calculate mean predicted measurement
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred += weights_(i)*Zsig.col(i);
  }

  //calculate measurement covariance matrix S
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_* std_laspx_, 0,
       0, std_laspy_*std_laspy_;

  S = S + R;

  //calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd y = meas_package.raw_measurements_ - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * y;
  P_ = P_ - K * S * K.transpose();

  // calculate normalized Innovation Squared (NIS)
  NIS_laser_ = y.transpose() * S.inverse() * y;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement. Lesson 7 Chapter 25 - 30.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  // number of measured values, i.e. r, phi, and r_dot
  int n_z = 3;

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    double r = sqrt(px*px + py*py);

    //check division by zero
    if (fabs(px) < 0.0001 || fabs(r) < 0.0001) {
      cout << "UpdateEKF () - Error - Division by Zero" << endl;
    }

    // measurement model
    Zsig(0, i) = r;                     //r
    Zsig(1, i) = atan2(py, px);         //phi
    Zsig(2, i) = (px*v1 + py*v2) / r;   //r_dot
  }

  //calculate mean predicted measurement
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred += weights_(i)*Zsig.col(i);
  }

  //calculate measurement covariance matrix S
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_*std_radr_, 0, 0,
       0, std_radphi_*std_radphi_, 0,
       0, 0, std_radrd_*std_radrd_;
  S = S + R;

  //cout << "radar measurement:" << endl << meas_package.raw_measurements_ << endl << endl;
  //cout << "radar prediction:" << endl << z_pred << endl << endl;
  //cout << "radar covariance:" << endl << S << endl << endl;

  //calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd y = meas_package.raw_measurements_ - z_pred;

  //angle normalization
  while (y(1)> M_PI) y(1) -= 2.*M_PI;
  while (y(1)<-M_PI) y(1) += 2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * y;
  P_ = P_ - K * S * K.transpose();

  // calculate normalized Innovation Squared (NIS)
  NIS_radar_ = y.transpose() * S.inverse() * y;
}

/**
* Generates Sigma Points for UKF. Lesson 7 Chapter 13 - 15.
*/
void UKF::GenerateSigmaPoints() {
  //calculate square root of P
  MatrixXd A = P_.llt().matrixL();

  //set first column of sigma point matrix
  Xsig_.col(0) = x_;

  //set remaining sigma points
  for (int i = 0; i < n_x_; i++)
  {
    Xsig_.col(i + 1) = x_ + sqrt(lambda_ + n_x_) * A.col(i);
    Xsig_.col(i + 1 + n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
  }
}

/**
* Augments Sigma Points for UKF, i.e.consider process noise as sigma points. Lesson 7 Chapter 16 - 18.
*/
void UKF::AugmentedSigmaPoints() {
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0.0);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);

  //create augmented mean state
  x_aug.head(n_x_) = x_;

  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_*std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_*std_yawdd_;

  //cout << "P_aug:" << endl << P_aug << endl << endl;
  
  //create square root matrix
  MatrixXd A_aug = P_aug.llt().matrixL();
  
  //cout << "A_aug:" << endl << A_aug << endl << endl;

  //create augmented sigma points
  Xsig_aug_.fill(0.0);
  Xsig_aug_.col(0) = x_aug;

  for (int j = 0; j < n_aug_; j++)
  {
    Xsig_aug_.col(j + 1)          = x_aug + sqrt(lambda_ + n_aug_) * A_aug.col(j);
    Xsig_aug_.col(j + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A_aug.col(j);
  }
}

/**
* Predict Sigma Points. Lesson 7 Chapter 19 - 21.
*/
void UKF::SigmaPointPrediction(double delta_t) {
  
  Xsig_pred_.fill(0.0);
  
  //predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug_(0, i);
    double p_y = Xsig_aug_(1, i);
    double v = Xsig_aug_(2, i);
    double yaw = Xsig_aug_(3, i);
    double yawd = Xsig_aug_(4, i);
    double nu_a = Xsig_aug_(5, i);
    double nu_yawdd = Xsig_aug_(6, i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
      py_p = p_y + v/yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
    }
    else {
      px_p = p_x + v*delta_t*cos(yaw);
      py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
}
