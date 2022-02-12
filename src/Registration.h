// -----------------------------------------------------------------------------------------------------------------------
// This work is based on Open3D-0.8.0, https://github.com/isl-org/Open3D/releases/tag/v0.8.0
// -----------------------------------------------------------------------------------------------------------------------

#include <eigen3/Eigen/Core>
#include <tuple>
#include <vector>

#include "Open3D/Registration/CorrespondenceChecker.h"
#include "Open3D/Registration/TransformationEstimation.h"
#include "Open3D/Utility/Eigen.h"

#include "Open3D/Geometry/KDTreeFlann.h"

namespace open3d {

namespace geometry {
class PointCloud;
}

namespace registration {

/// ICP algorithm stops if the relative change of fitness and rmse hit
/// relative_fitness_ and relative_rmse_ individually, or the iteration number
/// exceeds max_iteration_.
class ICPConvergenceCriteria {
public:
    ICPConvergenceCriteria(double relative_fitness = 1e-6,
                           double relative_rmse = 1e-6,
                           int max_iteration = 30)
        : relative_fitness_(relative_fitness),
          relative_rmse_(relative_rmse),
          max_iteration_(max_iteration) {}
    ~ICPConvergenceCriteria() {}

public:
    double relative_fitness_;
    double relative_rmse_;
    int max_iteration_;
};


RegistrationResult ImprovedRegistrationICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const RefinedTransformationEstimation &estimation,
        const Eigen::Matrix4d &init = Eigen::Matrix4d::Identity(),       
        const ICPConvergenceCriteria &criteria = ICPConvergenceCriteria());

}
}
