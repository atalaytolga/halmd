/*
 *
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

#include <halmd/mdsim/gpu/forces/external_kernel.cuh>
#include <halmd/mdsim/gpu/potentials/external/softcore_planar_wall_kernel.hpp>
#include <halmd/utility/tuple.hpp>

#include <cuda_wrapper/cuda_wrapper.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {
namespace potentials {
namespace external {
namespace softcore_planar_wall_kernel {

template <int dimension>
__device__ tuple<typename softcore_planar_wall<dimension>::vector_type, float>
softcore_planar_wall<dimension>::operator()(vector_type const& r) const
{
    float en_pot = 0;
    vector_type force = 0;

    // loop over walls
    for (unsigned int i = 0; i < nwall_; ++i) {

        // fetch geometry parameters from texture cache
        vector_type surface_normal;
        float offset;
        tie(surface_normal, offset) <<= tex1Dfetch<float4>(geometry_, i);

        // compute signed distance to wall i
        float d = inner_prod(r, surface_normal) - offset;

        // The normal points out of the accessible region. Apply the wall
        // only on the inner side of the plane.
        if (d >= 0)
            continue;

        float distance = -d;

        // fetch potential parameters from texture cache
        fixed_vector<float, 4> param =
            tex1Dfetch<float4>(potential_, species_ * nwall_ + i);

        float epsilon = param[EPSILON];
        float sigma = param[SIGMA];
        float cutoff = param[CUTOFF];

        // truncate interaction
        if (distance >= cutoff)
            continue;

        float lambda2 = lambda_ * lambda_;
        float soft_distance = distance + (1.0f - lambda_) * cutoff;

        float x = sigma / soft_distance;
        float x2 = x * x;
        float x6 = x2 * x2 * x2;
        float x12 = x6 * x6;

        float bracket = x12 - x6 + 0.25f;

        float delta = distance - cutoff;
        float delta2 = delta * delta;
        float delta3 = delta2 * delta;
        float delta4 = delta2 * delta2;

        float h2 = smoothing_ * smoothing_;
        float h4 = h2 * h2;

        float denominator = delta4 + h4;
        float cutoff_switch = delta4 / denominator;

        float switch_derivative =
            4.0f * h4 * delta3 / (denominator * denominator);

        float energy_raw = 4.0f * epsilon * lambda2 * bracket;
        float energy = energy_raw * cutoff_switch;

        float force_core =
            24.0f * epsilon * lambda2 * (2.0f * x12 - x6)
            / soft_distance;

        float force_magnitude =
            force_core * cutoff_switch
            - energy_raw * switch_derivative;

        // accumulate force and potential energy
        force -= force_magnitude * surface_normal;
        en_pot += energy;
    }
    return make_tuple(force, en_pot);
}

} // namespace softcore_planar_wall_kernel

} // namespace external
} // namespace potentials

// explicit instantiation of force kernels
namespace forces {

using namespace potentials::external::softcore_planar_wall_kernel;

template class external_wrapper<3, softcore_planar_wall<3> >;
template class external_wrapper<2, softcore_planar_wall<2> >;

} // namespace forces

} // namespace gpu
} // namespace mdsim
} // namespace halmd
