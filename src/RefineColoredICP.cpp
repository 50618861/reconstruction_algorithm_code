// -----------------------------------------------------------------------------------------------------------------------
// This work is based on Open3D-0.8.0, https://github.com/isl-org/Open3D/releases/tag/v0.8.0
// -----------------------------------------------------------------------------------------------------------------------

#include "Open3D/Registration/RefineColoredICP.h"
#include "Open3D/Registration/Registration.h"

#include <eigen3/Eigen/Dense>
#include <iostream>

#include "Open3D/Geometry/KDTreeFlann.h"
#include "Open3D/Geometry/KDTreeSearchParam.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/Eigen.h"

namespace open3d {

namespace {
using namespace registration;

class PointCloudForColoredICP : public geometry::PointCloud {
public:
    std::vector<Eigen::Vector3d> color_gradient_;
};

class RefinedTransformationEstimationForColoredICP : public RefinedTransformationEstimation {
public:
    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    RefinedTransformationEstimationForColoredICP(double lambda_geometric = 0.968)
        : lambda_geometric_(lambda_geometric) {
        if (lambda_geometric_ < 0 || lambda_geometric_ > 1.0)
            lambda_geometric_ = 0.968;
    }
    ~RefinedTransformationEstimationForColoredICP() override {}

public:
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const CorrespondenceSet &corres) const override;

    std::tuple<Eigen::Matrix4d, double, double> ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres,
            double max_correspondence_distance,
            int k,
            double mu,
            double nu) const override;

public:
    double lambda_geometric_;
private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::refinedColoredICP;
};

std::shared_ptr<PointCloudForColoredICP> InitializePointCloudForColoredICP(
        const geometry::PointCloud &target,
        const geometry::KDTreeSearchParamHybrid &search_param) {
    utility::LogDebug("InitializePointCloudForColoredICP\n");

    geometry::KDTreeFlann tree;
    tree.SetGeometry(target);

    auto output = std::make_shared<PointCloudForColoredICP>();
    output->colors_ = target.colors_;
    output->normals_ = target.normals_;
    output->points_ = target.points_;

    size_t n_points = output->points_.size();
    output->color_gradient_.resize(n_points, Eigen::Vector3d::Zero());

    for (size_t k = 0; k < n_points; k++) {
        const Eigen::Vector3d &vt = output->points_[k];
        const Eigen::Vector3d &nt = output->normals_[k];

        // compute the grayscale rather than average colour intensity.
        double it = 0.299 * output->colors_[k](0)  + 0.587 * output->colors_[k](1) + 0.114*output->colors_[k](2);

        std::vector<int> point_idx;
        std::vector<double> point_squared_distance;

        if (tree.SearchHybrid(vt, search_param.radius_, search_param.max_nn_,
                              point_idx, point_squared_distance) >= 3) {
            // approximate image gradient of vt's tangential plane
            size_t nn = point_idx.size();
            Eigen::MatrixXd A(nn, 3);
            Eigen::MatrixXd b(nn, 1);
            A.setZero();
            b.setZero();
            for (size_t i = 1; i < nn; i++) {
                int P_adj_idx = point_idx[i];
                Eigen::Vector3d vt_adj = output->points_[P_adj_idx];
                Eigen::Vector3d vt_proj = vt_adj - (vt_adj - vt).dot(nt) * nt;

                double it_adj =  0.299 * output->colors_[P_adj_idx](0) +
                                 0.587 * output->colors_[P_adj_idx](1) +
                                 0.114 * output->colors_[P_adj_idx](2);

                A(i - 1, 0) = (vt_proj(0) - vt(0));
                A(i - 1, 1) = (vt_proj(1) - vt(1));
                A(i - 1, 2) = (vt_proj(2) - vt(2));
                b(i - 1, 0) = (it_adj - it);
            }
            // adds orthogonal constraint
            A(nn - 1, 0) = (nn - 1) * nt(0);
            A(nn - 1, 1) = (nn - 1) * nt(1);
            A(nn - 1, 2) = (nn - 1) * nt(2);
            b(nn - 1, 0) = 0;
            // solving linear equation
            bool is_success;
            Eigen::MatrixXd x;
            std::tie(is_success, x) = utility::SolveLinearSystemPSD(
                    A.transpose() * A, A.transpose() * b);
            if (is_success) {
                output->color_gradient_[k] = x;
            }
        }
    }
    return output;
}

std::tuple<Eigen::Matrix4d, double, double> RefinedTransformationEstimationForColoredICP::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        double max_correspondence_distance,
        int k,
        double mu,
        double nu) const {
    if (corres.empty() || target.HasNormals() == false ||
        target.HasColors() == false || source.HasColors() == false)
        return std::make_tuple(Eigen::Matrix4d::Identity(), mu, nu);

    double sqrt_lambda_geometric = sqrt(lambda_geometric_);
    double lambda_photometric = 1.0 - lambda_geometric_;
    double sqrt_lambda_photometric = sqrt(lambda_photometric);

    const auto &target_c = (const PointCloudForColoredICP &)target;

    auto compute_jacobian_and_residual =
            [&](int i,
                std::vector<Eigen::Vector6d, utility::Vector6d_allocator> &J_r,
                std::vector<double> &r) {
                size_t cs = corres[i][0];
                size_t ct = corres[i][1];
                const Eigen::Vector3d &vs = source.points_[cs];
                const Eigen::Vector3d &vt = target.points_[ct];
                const Eigen::Vector3d &nt = target.normals_[ct];

                J_r.resize(2);
                r.resize(2);

                J_r[0].block<3, 1>(0, 0) = sqrt_lambda_geometric * vs.cross(nt);
                J_r[0].block<3, 1>(3, 0) = sqrt_lambda_geometric * nt;
                r[0] = sqrt_lambda_geometric * (vs - vt).dot(nt);

                // project vs into vt's tangential plane
                Eigen::Vector3d vs_proj = vs - (vs - vt).dot(nt) * nt;

                double is = 0.299 * source.colors_[cs](0) + 
                            0.587 * source.colors_[cs](1) +
                            0.114 * source.colors_[cs](2);

                double it = 0.299 * target.colors_[ct](0) + 
                            0.587 * target.colors_[ct](1) +
                            0.114 * target.colors_[ct](2);
                
                const Eigen::Vector3d &dit = target_c.color_gradient_[ct];
                double is0_proj = (dit.dot(vs_proj - vt)) + it;

                const Eigen::Matrix3d M =
                        (Eigen::Matrix3d() << 1.0 - nt(0) * nt(0),
                         -nt(0) * nt(1), -nt(0) * nt(2), -nt(0) * nt(1),
                         1.0 - nt(1) * nt(1), -nt(1) * nt(2), -nt(0) * nt(2),
                         -nt(1) * nt(2), 1.0 - nt(2) * nt(2))
                                .finished();

                const Eigen::Vector3d &ditM = -dit.transpose() * M;
                J_r[1].block<3, 1>(0, 0) =
                        sqrt_lambda_photometric * vs.cross(ditM);
                J_r[1].block<3, 1>(3, 0) = sqrt_lambda_photometric * ditM;
                r[1] = sqrt_lambda_photometric * (is - is0_proj);
            };

    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
    Eigen::Vector6d J;

    double r2;

    std::tie(JTJ, JTr, J, r2) =
        utility::ComputeJTJLandJTr<Eigen::Matrix6d, Eigen::Vector6d>(
                compute_jacobian_and_residual, (int)corres.size());    

    if (k == 0)
    {      
        std::vector<double> B;
        B.push_back( JTJ(0, 0) );
        B.push_back( JTJ(1, 1) );
        B.push_back( JTJ(2, 2) ); 
        auto max_p = std::max_element(B.begin(), B.end());
        mu = *max_p;
    }

    bool is_success;
    Eigen::Matrix4d extrinsic;
    Eigen::Vector6d x_;

    // Levenberg-Marquardt method to solve increment x_ (in thesis, we use \xi to represent)  
    std::tie(is_success, extrinsic, x_) =
            utility::LMObtainExtrinsicMatrix(JTJ, JTr, mu);

    Eigen::MatrixXd L = -x_.transpose() * J * sqrt(r2) - 0.5 * x_.transpose() * J * J.transpose() * x_;

    double l = L.determinant();
    geometry::PointCloud pcd = source;
    pcd.Transform(extrinsic);

    auto source_c = InitializePointCloudForColoredICP(
            source, geometry::KDTreeSearchParamHybrid(0.1, 30));
    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(*source_c);
    
    registration::RegistrationResult result = GetRegistrationResultAndCorrespondences(pcd, *source_c, kdtree, max_correspondence_distance, extrinsic);
    double fx_fxd;
    fx_fxd = ComputeRMSE(pcd, *source_c, result.correspondence_set_);
    double rho = -fx_fxd / l;
    if( rho > 0)
    {
        mu = mu * std::max<double>(0.33, 1 - std::abs(std::pow(2*rho -1, 3)));
        // mu = mu * 0.33;
        nu = 2.0;
    }
    else
    {
        mu = mu * nu; 
        nu = 2*nu;
    }

    if (is_success) 
    {
        return std::make_tuple(std::move(extrinsic), mu, nu);
    }
    else
    {
        return std::make_tuple(Eigen::Matrix4d::Identity(), mu, nu);
    }
}

double RefinedTransformationEstimationForColoredICP::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    double sqrt_lambda_geometric = sqrt(lambda_geometric_);
    double lambda_photometric = 1.0 - lambda_geometric_;
    double sqrt_lambda_photometric = sqrt(lambda_photometric);
    const auto &target_c = (const PointCloudForColoredICP &)target;

    double residual = 0.0;
    for (size_t i = 0; i < corres.size(); i++) {
        size_t cs = corres[i][0];
        size_t ct = corres[i][1];
        const Eigen::Vector3d &vs = source.points_[cs];
        const Eigen::Vector3d &vt = target.points_[ct];
        const Eigen::Vector3d &nt = target.normals_[ct];
        Eigen::Vector3d vs_proj = vs - (vs - vt).dot(nt) * nt;

        double is = 0.299 * source.colors_[cs](0) + 
                    0.587 * source.colors_[cs](1) +
                    0.114 * source.colors_[cs](2);

        double it = 0.299 * target.colors_[ct](0) + 
                    0.587 * target.colors_[ct](1) +
                    0.114 * target.colors_[ct](2);

        const Eigen::Vector3d &dit = target_c.color_gradient_[ct];

        double is0_proj = (dit.dot(vs_proj - vt)) + it;        
        double residual_geometric = sqrt_lambda_geometric * (vs - vt).dot(nt);
        double residual_photometric = sqrt_lambda_photometric * (is - is0_proj);
        residual += residual_geometric * residual_geometric +
                    residual_photometric * residual_photometric;
    }
    return residual;
};

}  // unnamed namespace

namespace registration {

RegistrationResult RefineRegistrationColoredICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_distance,
        const Eigen::Matrix4d &init /* = Eigen::Matrix4d::Identity()*/,
        const ICPConvergenceCriteria &criteria /* = ICPConvergenceCriteria()*/,
        double lambda_geometric /* = 0.968*/) 
        
    {
    auto target_c = InitializePointCloudForColoredICP(
            target, geometry::KDTreeSearchParamHybrid(max_distance * 2.0, 30));
    

    return ImprovedRegistrationICP(
            source, *target_c, max_distance,
            RefinedTransformationEstimationForColoredICP(lambda_geometric), init, criteria);
}

}  // namespace registration
}  // namespace open3d
