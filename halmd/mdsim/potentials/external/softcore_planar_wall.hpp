/*
 * Copyright © 2026      Tolga Atalay
 *
 * This file is part of HALMD.
 *
 * HALMD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * HALMD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General
 * Public License along with HALMD. If not, see
 * <http://www.gnu.org/licenses/>.
 */

#ifndef HALMD_MDSIM_POTENTIALS_EXTERNAL_SOFTCORE_PLANAR_WALL_HPP
#define HALMD_MDSIM_POTENTIALS_EXTERNAL_SOFTCORE_PLANAR_WALL_HPP

#include <stdexcept>

#include <halmd/numeric/blas/fixed_vector.hpp>

namespace halmd {
namespace mdsim {
namespace potentials {
namespace external {
namespace detail {

/**
 * Evaluate the coupling derivative at fixed position using host-side parameters.
 * Shared by the host and GPU potential objects for their Lua point queries.
 */
template <typename potential_type>
typename potential_type::vector_type::value_type softcore_planar_wall_du_dlambda(
    potential_type const& potential
  , typename potential_type::vector_type const& r
  , unsigned int species
)
{
    typedef typename potential_type::vector_type::value_type float_type;
    if (species >= potential.size()) {
        throw std::invalid_argument("particle species is out of range");
    }

    float_type derivative = 0;
    float_type const lambda = potential.lambda();
    float_type const h2 = potential.smoothing() * potential.smoothing();
    float_type const h4 = h2 * h2;

    for (unsigned int i = 0; i < potential.surface_normal().size(); ++i) {
        float_type const d = inner_prod(r, potential.surface_normal()(i)) - potential.offset()(i);
        float_type const cutoff = potential.cutoff()(i, species);
        if (d >= 0 || -d >= cutoff) {
            continue;
        }

        float_type const distance = -d;
        float_type const soft_distance = distance + (1 - lambda) * cutoff;
        float_type const x = potential.sigma()(i, species) / soft_distance;
        float_type const x2 = x * x;
        float_type const x6 = x2 * x2 * x2;
        float_type const x12 = x6 * x6;
        float_type const bracket = x12 - x6 + float_type(0.25);
        float_type const delta = distance - cutoff;
        float_type const delta2 = delta * delta;
        float_type const delta4 = delta2 * delta2;
        float_type const cutoff_switch = delta4 / (delta4 + h4);
        float_type const epsilon = potential.epsilon()(i, species);
        float_type const force_core = 24 * epsilon * lambda * lambda
            * (2 * x12 - x6) / soft_distance;

        // The switch depends on distance only, and d(soft_distance)/d(lambda) = -cutoff.
        derivative += cutoff_switch * (8 * epsilon * lambda * bracket + cutoff * force_core);
    }
    return derivative;
}

} // namespace detail
} // namespace external
} // namespace potentials
} // namespace mdsim
} // namespace halmd

#endif /* ! HALMD_MDSIM_POTENTIALS_EXTERNAL_SOFTCORE_PLANAR_WALL_HPP */
