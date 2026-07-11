/*
 * Copyright © 2026 Tolga Atalay
 * Copyright © 2010, 2012 Peter Colberg
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

#ifndef HALMD_MDSIM_GPU_POSITION_SHIFT_HPP
#define HALMD_MDSIM_GPU_POSITION_SHIFT_HPP

#include <halmd/mdsim/box.hpp>
#include <halmd/mdsim/gpu/particle_group.hpp>
#include <halmd/mdsim/gpu/position_shift_kernel.hpp>
#include <halmd/numeric/blas/fixed_vector.hpp>
#include <halmd/utility/gpu/configure_kernel.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {

/**
 * Shift all positions by 'delta'
 */
template <typename particle_type>
inline void shift_position(
    particle_type& particle
  , box<particle_type::position_type::static_size> const& box
  , fixed_vector<double, particle_type::position_type::static_size> const& delta
)
{
    typedef typename particle_type::position_type position_type;

    auto position = make_cache_mutable(particle.position());
    auto image = make_cache_mutable(particle.image());

    position_type const box_length =
        static_cast<position_type>(box.length());

    auto& kernel =
        get_position_shift_kernel<
        particle_type::position_type::static_size
      , typename particle_type::float_type
      >().shift;

    configure_kernel(kernel, particle.dim(), false);
    kernel(
        position->data()
      , image->data()
      , particle.nparticle()
      , particle.dim().threads()
      , delta
      , box_length
    );
}

/**
 * Shift positions of particle group by 'delta'
 */
template <typename particle_type>
inline void shift_position_group(
    particle_type& particle
  , particle_group& group
  , box<particle_type::position_type::static_size> const& box
  , fixed_vector<double, particle_type::position_type::static_size> const& delta
)
{
    typedef typename particle_type::position_type position_type;

    auto const& unordered = read_cache(group.unordered());
    auto position = make_cache_mutable(particle.position());
    auto image = make_cache_mutable(particle.image());

    position_type const box_length =
        static_cast<position_type>(box.length());

    auto& kernel =
        get_position_shift_kernel<
        particle_type::position_type::static_size
      , typename particle_type::float_type
      >().shift_group;

    configure_kernel(kernel, unordered.size());
    kernel(
        position->data()
      , image->data()
      , unordered.data()
      , unordered.size()
      , particle.dim().threads()
      , delta
      , box_length
    );
}

} // namespace gpu
} // namespace mdsim
} // namespace halmd

#endif /* ! HALMD_MDSIM_GPU_POSITION_SHIFT_HPP */
