// -----------------------------------------------------------------------------------------------------------------------
// This work is based on Open3D-0.8.0, https://github.com/isl-org/Open3D/releases/tag/v0.8.0
// -----------------------------------------------------------------------------------------------------------------------


#include <eigen3/Eigen/Core>

#include "Open3D/Registration/Registration.h"

namespace open3d {

namespace geometry {
class PointCloud;
}

namespace registration {
class RegistrationResult;

RegistrationResult RefineRegistrationColoredICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_distance,
        const Eigen::Matrix4d &init = Eigen::Matrix4d::Identity(),
        const ICPConvergenceCriteria &criteria = ICPConvergenceCriteria(),
        double lambda_geometric = 0.968);

} 
} 
