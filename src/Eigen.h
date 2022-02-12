// -----------------------------------------------------------------------------------------------------------------------
// This work is based on Open3D-0.8.0, https://github.com/isl-org/Open3D/releases/tag/v0.8.0
// -----------------------------------------------------------------------------------------------------------------------

#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/StdVector>
#include <tuple>
#include <vector>

namespace Eigen {

/// Extending Eigen namespace by adding frequently used matrix type
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

/// Use Eigen::DontAlign for matrices inside classes which are exposed in the
/// Open3D headers https://github.com/intel-isl/Open3D/issues/653
typedef Eigen::Matrix<double, 6, 6, Eigen::DontAlign> Matrix6d_u;
typedef Eigen::Matrix<double, 4, 4, Eigen::DontAlign> Matrix4d_u;

}  // namespace Eigen

namespace open3d {
namespace utility {


/// Function to transform 6D motion vector to 4D motion matrix
/// Reference:
/// https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html#TutorialGeoTransform
Eigen::Matrix4d TransformVector6dToMatrix4d(const Eigen::Vector6d &input);


/// Function to solve Ax + \mu=b

std::tuple<bool, Eigen::VectorXd> Levenberg_Marquardt(
        const Eigen::MatrixXd &A,
        const Eigen::VectorXd &b,
        const double &mu,
        bool prefer_sparse = false,
        bool check_symmetric = false,
        bool check_det = false,
        bool check_psd = false);
/// Function to solve Jacobian system
/// Input: 6x6 Jacobian matrix and 6-dim residual vector.
/// Output: tuple of is_success, 4x4 extrinsic matrices.;
std::tuple<bool, Eigen::Matrix4d, Eigen::Vector6d> LMObtainExtrinsicMatrix(
        const Eigen::Matrix6d &JTJ, const Eigen::Vector6d &JTr, const double &mu);

/// Function to compute JTJ and Jtr
/// Input: function pointer f and total number of rows of Jacobian matrix
/// Output: JTJ, JTr, sum of r^2
/// Note: f takes index of row, and outputs corresponding residual and row
/// vector.

template <typename MatType, typename VecType>
std::tuple<MatType, VecType, VecType, double> ComputeJTJLandJTr(
        std::function<
                void(int,
                     std::vector<VecType, Eigen::aligned_allocator<VecType>> &,
                     std::vector<double> &)> f,
        int iteration_num,
        bool verbose = true);


}  // namespace utility
}  // namespace open3d
