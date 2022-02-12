// -----------------------------------------------------------------------------------------------------------------------
// This work is based on Open3D-0.8.0, https://github.com/isl-org/Open3D/releases/tag/v0.8.0
// -----------------------------------------------------------------------------------------------------------------------

#include "Open3D/Utility/Eigen.h"

#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Sparse>

#include "Open3D/Utility/Console.h"

namespace open3d {
namespace utility {


/// New Method to solve Ax+\mu=b
std::tuple<bool, Eigen::VectorXd> Levenberg_Marquardt(
        const Eigen::MatrixXd &A,
        const Eigen::VectorXd &b,
        const double &mu,
        bool prefer_sparse /* = false */,
        bool check_symmetric /* = false */,
        bool check_det /* = false */,
        bool check_psd /* = false */) {
    // PSD implies symmetric
    check_symmetric = check_symmetric || check_psd;
    if (check_symmetric && !A.isApprox(A.transpose())) {
        LogWarning("check_symmetric failed, empty vector will be returned\n");
        return std::make_tuple(false, Eigen::VectorXd::Zero(b.rows()));
    }

    if (check_det) {
        double det = A.determinant();
        if (fabs(det) < 1e-6 || std::isnan(det) || std::isinf(det)) {
            LogWarning("check_det failed, empty vector will be returned\n");
            return std::make_tuple(false, Eigen::VectorXd::Zero(b.rows()));
        }
    }

    // Check PSD: https://stackoverflow.com/a/54569657/1255535
    if (check_psd) {
        Eigen::LLT<Eigen::MatrixXd> A_llt(A);
        if (A_llt.info() == Eigen::NumericalIssue) {
            LogWarning("check_psd failed, empty vector will be returned\n");
            return std::make_tuple(false, Eigen::VectorXd::Zero(b.rows()));
        }
    }

    Eigen::VectorXd x(b.size());

    if (prefer_sparse) {
        Eigen::SparseMatrix<double> A_sparse = A.sparseView();
        // TODO: avoid deprecated API SimplicialCholesky
        Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> A_chol;
        A_chol.compute(A_sparse);
        if (A_chol.info() == Eigen::Success) {
            x = A_chol.solve(b);
            if (A_chol.info() == Eigen::Success) {
                // Both decompose and solve are successful
                return std::make_tuple(true, std::move(x));
            } else {
                LogWarning("Cholesky solve failed, switched to dense solver\n");
            }
        } else {
            LogWarning("Cholesky decompose failed, switched to dense solver\n");
        }
    }

    Eigen::Matrix6d G = A + mu * Eigen::Matrix6d::Identity();
    x = G.ldlt().solve(b);
    // x = G.colPivHouseholderQr().solve(b);

    return std::make_tuple(true, std::move(x));
}


Eigen::Matrix4d TransformVector6dToMatrix4d(const Eigen::Vector6d &input) {
    Eigen::Matrix4d output;
    output.setIdentity();
    output.block<3, 3>(0, 0) =
            (Eigen::AngleAxisd(input(2), Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(input(1), Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(input(0), Eigen::Vector3d::UnitX()))
                    .matrix();
    output.block<3, 1>(0, 3) = input.block<3, 1>(3, 0);
    return output;
}



std::tuple<bool, Eigen::Matrix4d, Eigen::Vector6d> LMObtainExtrinsicMatrix(
        const Eigen::Matrix6d &JTJ, const Eigen::Vector6d &JTr, const double &mu) {
    std::vector<Eigen::Matrix4d, Matrix4d_allocator> output_matrix_array;
    output_matrix_array.clear();

    bool solution_exist;
    Eigen::Vector6d x;
    std::tie(solution_exist, x) = Levenberg_Marquardt(JTJ, -JTr, mu);

    if (solution_exist) {
        Eigen::Matrix4d extrinsic = TransformVector6dToMatrix4d(x);
        return std::make_tuple(solution_exist, std::move(extrinsic),std::move(x));
    } else {
        return std::make_tuple(false, Eigen::Matrix4d::Identity(), Eigen::Vector6d::Identity());
    }
}



template <typename MatType, typename VecType>
std::tuple<MatType, VecType, VecType, double> ComputeJTJLandJTr(
        std::function<
                void(int,
                     std::vector<VecType, Eigen::aligned_allocator<VecType>> &,
                     std::vector<double> &)> f,
        int iteration_num,
        bool verbose /*=true*/) {
    MatType JTJ;
    VecType J;
    VecType JTr;
    double r2_sum = 0.0;
    JTJ.setZero();
    JTr.setZero();
    J.setZero();
#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        MatType JTJ_private;
        VecType J_private;
        VecType JTr_private;
        double r2_sum_private = 0.0;
        JTJ_private.setZero();
        J_private.setZero();
        JTr_private.setZero();
        std::vector<double> r;
        std::vector<VecType, Eigen::aligned_allocator<VecType>> J_r;
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int i = 0; i < iteration_num; i++) {
            f(i, J_r, r);
            for (int j = 0; j < (int)r.size(); j++) {
                JTJ_private.noalias() += J_r[j] * J_r[j].transpose();
                JTr_private.noalias() += J_r[j] * r[j];
                J_private.noalias() += J_r[j];
                r2_sum_private += r[j] * r[j];
            }
        }
#ifdef _OPENMP
#pragma omp critical
        {
#endif
            JTJ += JTJ_private;
            JTr += JTr_private;
            J += J_private;
            r2_sum += r2_sum_private;
#ifdef _OPENMP
        }
    }
#endif
    if (verbose) {
        LogDebug("Residual : {:.2e} (# of elements : {:d})\n",
                 r2_sum / (double)iteration_num, iteration_num);
    }
    return std::make_tuple(std::move(JTJ), std::move(JTr), std::move(J), r2_sum);
}



template std::tuple<Eigen::Matrix6d, Eigen::Vector6d, Eigen::Vector6d, double> ComputeJTJLandJTr(
        std::function<void(int,
                           std::vector<Eigen::Vector6d, Vector6d_allocator> &,
                           std::vector<double> &)> f,
        int iteration_num, bool verbose);


}  // namespace utility
}  // namespace open3d
