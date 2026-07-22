/*
 * Copyright © 2026      Tolga Atalay
 * Copyright © 2014-2015 Sutapa Roy
 * Copyright © 2014-2015 Felix Höfling
 * Copyright © 2020      Jaslo Ziska
 *
 * This file is part of HALMD.
 *
 * HALMD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General
 * Public License along with this program. If not, see
 * <http://www.gnu.org/licenses/>.
 */

#ifndef HALMD_MDSIM_GPU_POTENTIALS_EXTERNAL_SOFTCORE_PLANAR_WALL_KERNEL_HPP
#define HALMD_MDSIM_GPU_POTENTIALS_EXTERNAL_SOFTCORE_PLANAR_WALL_KERNEL_HPP

#include <halmd/utility/tuple.hpp>

#include <cuda_wrapper/cuda_wrapper.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {
namespace potentials {
namespace external {
namespace softcore_planar_wall_kernel {

/**
 * indices of potential parameters
 */
enum {
    EPSILON     /* interaction strength */
  , SIGMA       /* interaction range */
  , CUTOFF      /* cutoff distance */
};

/**
 * Soft-core planar wall external potential.
 */
template <int dimension>
class softcore_planar_wall
{
public:
    typedef fixed_vector<float, dimension> vector_type;

    /**
     * Construct soft-core planar wall potential.
     */
    HALMD_GPU_ENABLED softcore_planar_wall(
        cudaTextureObject_t geometry
      , cudaTextureObject_t potential
      , unsigned int nwall
      , float smoothing
      , float lambda
    )
      : species_(0)
      , geometry_(geometry)
      , potential_(potential)
      , nwall_(nwall)
      , smoothing_(smoothing)
      , lambda_(lambda)
    {}

    /**
     * Fetch parameters for this particle species.
     */
    HALMD_GPU_ENABLED void fetch_param(unsigned int species)
    {
        species_ = species;
    }

    /**
     * Compute force and potential energy due to all planar walls.
     */
    HALMD_GPU_ENABLED tuple<vector_type, float>
    operator()(vector_type const& r) const;

private:
    unsigned int species_;
    cudaTextureObject_t geometry_;
    cudaTextureObject_t potential_;
    unsigned int nwall_;
    float smoothing_;
    float lambda_;
};

} // namespace softcore_planar_wall_kernel

} // namespace external
} // namespace potentials
} // namespace gpu
} // namespace mdsim
} // namespace halmd

#endif /* ! HALMD_MDSIM_GPU_POTENTIALS_EXTERNAL_SOFTCORE_PLANAR_WALL_KERNEL_HPP */
