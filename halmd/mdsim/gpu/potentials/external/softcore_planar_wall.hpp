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

#ifndef HALMD_MDSIM_GPU_POTENTIALS_EXTERNAL_SOFTCORE_PLANAR_WALL_HPP
#define HALMD_MDSIM_GPU_POTENTIALS_EXTERNAL_SOFTCORE_PLANAR_WALL_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <lua.hpp>
#include <memory>

#include <halmd/io/logger.hpp>
#include <halmd/mdsim/gpu/potentials/external/softcore_planar_wall_kernel.hpp>
#include <halmd/numeric/blas/fixed_vector.hpp>
#include <halmd/utility/signal.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {
namespace potentials {
namespace external {

/**
 * Define soft-core planar wall potential and parameters.
 */
template <int dimension, typename float_type>
class softcore_planar_wall
{
public:
    typedef halmd::signal<void ()> signal_type;
    typedef signal_type::slot_function_type slot_function_type;

    typedef softcore_planar_wall_kernel::softcore_planar_wall<dimension> gpu_potential_type;

    typedef fixed_vector<float_type, dimension> vector_type;
    typedef boost::numeric::ublas::vector<float_type> scalar_container_type;
    typedef boost::numeric::ublas::vector<vector_type> vector_container_type;
    typedef boost::numeric::ublas::matrix<float_type> matrix_container_type;

    softcore_planar_wall(
        scalar_container_type const& offset
      , vector_container_type const& surface_normal
      , matrix_container_type const& epsilon
      , matrix_container_type const& sigma
      , matrix_container_type const& cutoff
      , float_type smoothing
      , float_type lambda
      , std::shared_ptr<halmd::logger> logger = std::make_shared<halmd::logger>()
    );

    /**
     * Update planar wall positions.
     */
    void set_offset(scalar_container_type const& offset);

    /**
     * Connect slot to wall update signal.
     */
    connection on_set_offset(slot_function_type const& slot)
    {
        return on_set_offset_.connect(slot);
    }

    /** return gpu potential with textures */
    gpu_potential_type get_gpu_potential() const
    {
        return gpu_potential_type(
            t_param_geometry_
          , t_param_potential_
          , static_cast<unsigned int>(surface_normal_.size())
          , static_cast<float>(smoothing_)
          , static_cast<float>(lambda_)
        );
    }

    scalar_container_type const& offset() const
    {
        return offset_;
    }

    vector_container_type const& surface_normal() const
    {
        return surface_normal_;
    }

    matrix_container_type const& epsilon() const
    {
        return epsilon_;
    }

    matrix_container_type const& sigma() const
    {
        return sigma_;
    }

    matrix_container_type const& cutoff() const
    {
        return cutoff_;
    }

    float_type smoothing() const
    {
        return smoothing_;
    }

    float_type lambda() const
    {
        return lambda_;
    }

    // size of parameter arrays, must match number of particle species
    unsigned int size() const
    {
        return epsilon_.size2();
    }

    /**
     * Bind class to Lua.
     */
    static void luaopen(lua_State* L);

private:
    /** wall positions in MD units */
    scalar_container_type offset_;
    /** wall normal vectors in MD units */
    vector_container_type surface_normal_;
    /** interaction strengths indexed by wall and particle species */
    matrix_container_type epsilon_;
    /** interaction ranges indexed by wall and particle species */
    matrix_container_type sigma_;
    /** cutoff distances indexed by wall and particle species */
    matrix_container_type cutoff_;
    /** smoothing parameter for potential smoothing in MD units */
    float_type smoothing_;
    /** coupling parameter */
    float_type lambda_;

    /** potential parameters at CUDA device */
    cuda::memory::device::vector<float4> g_param_geometry_;
    cuda::memory::device::vector<float4> g_param_potential_;

    /** array of geometry parameters for all walls */
    cuda::texture<float4> t_param_geometry_;
    /** array of potential parameters for all particle species */
    cuda::texture<float4> t_param_potential_;

    /** module logger */
    std::shared_ptr<logger> logger_;

    /** wall offset update signal */
    signal_type on_set_offset_;
};

} // namespace external
} // namespace potentials
} // namespace gpu
} // namespace mdsim
} // namespace halmd

#endif /* ! HALMD_MDSIM_GPU_POTENTIALS_EXTERNAL_SOFTCORE_PLANAR_WALL_HPP */
