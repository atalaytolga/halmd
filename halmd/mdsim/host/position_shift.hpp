/*
 * Copyright © 2026 Tolga Atalay
 * Copyright © 2010, 2012 Peter Colberg
 * Copyright © 2010 Felix Höfling
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

#ifndef HALMD_MDSIM_HOST_POSITION_SHIFT_HPP
#define HALMD_MDSIM_HOST_POSITION_SHIFT_HPP

#include <halmd/mdsim/box.hpp>
#include <halmd/mdsim/host/particle_group.hpp>
#include <halmd/numeric/blas/fixed_vector.hpp>

namespace halmd {
namespace mdsim {
namespace host {

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
    typedef typename particle_type::image_type image_type;

    auto position = make_cache_mutable(particle.position());
    auto image = make_cache_mutable(particle.image());

    for (unsigned int i = 0; i < particle.nparticle(); ++i) {
        auto& r = (*position)[i];
        auto& img = (*image)[i];

        r += delta;

        image_type crossing;
        while ((crossing = box.reduce_periodic(r)) != image_type(0)) {
            img += crossing;
        }
    }
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
    typedef typename particle_type::image_type image_type;

    auto const& unordered = read_cache(group.unordered());

    auto position = make_cache_mutable(particle.position());
    auto image = make_cache_mutable(particle.image());

    for (typename particle_group::size_type i : unordered) {
        auto& r = (*position)[i];
        auto& img = (*image)[i];

        r += delta;

        image_type crossing;
        while ((crossing = box.reduce_periodic(r)) != image_type(0)) {
            img += crossing;
        }
    }
}

} // namespace host
} // namespace mdsim
} // namespace halmd

#endif /* ! HALMD_MDSIM_HOST_POSITION_SHIFT_HPP */
