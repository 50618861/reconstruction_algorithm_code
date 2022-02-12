// -----------------------------------------------------------------------------------------------------------------------
// This work is based on Open3D-0.8.0, https://github.com/isl-org/Open3D/releases/tag/v0.8.0
// -----------------------------------------------------------------------------------------------------------------------
#include "Open3D/Registration/Registration.h"

#include <cstdlib>
#include <ctime>

#include "Open3D/Geometry/KDTreeFlann.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Registration/Feature.h"
#include "Open3D/Utility/Console.h"

namespace open3d {

namespace {
using namespace registration;

RegistrationResult ImprovedRegistrationICP (
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const RefinedTransformationEstimation &estimation,
        const Eigen::Matrix4d &init /* = Eigen::Matrix4d::Identity()*/,      
        /* = TransformationEstimationPointToPoint(false)*/
        const ICPConvergenceCriteria
                &criteria /* = ICPConvergenceCriteria()*/) {
    if (max_correspondence_distance <= 0.0) {
        utility::LogWarning("Invalid max_correspondence_distance.\n");
        return RegistrationResult(init);
    }

    Eigen::Matrix4d transformation = init;
    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(target);

    geometry::PointCloud pcd = source;
    if (init.isIdentity() == false) {
        pcd.Transform(init);
    }
    RegistrationResult result;
    result = GetRegistrationResultAndCorrespondences(
            pcd, target, kdtree, max_correspondence_distance, transformation);
    double nu = 2.0;
    double mu = 2.0;
    Eigen::Matrix4d update;
    for (int i = 0; i < criteria.max_iteration_; i++) {
        utility::LogDebug("ICP Iteration #{:d}: Fitness {:.4f}, RMSE {:.4f}\n",
                          i, result.fitness_, result.inlier_rmse_);
        std::tie (update,mu, nu)  = estimation.ComputeTransformation(
                pcd, target, result.correspondence_set_,max_correspondence_distance,i,mu,nu);
        transformation = update * transformation;
        pcd.Transform(update);
        RegistrationResult backup = result;
        result = GetRegistrationResultAndCorrespondences(
                pcd, target, kdtree, max_correspondence_distance,
                transformation);
        if (std::abs(backup.fitness_ - result.fitness_) <
                    criteria.relative_fitness_ &&
            std::abs(backup.inlier_rmse_ - result.inlier_rmse_) <
                    criteria.relative_rmse_) {
            break;
        }
    }
    return result;
}
