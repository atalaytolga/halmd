/*
 * Copyright © 2026 Tolga Atalay
 * Copyright © 2008-2010, 2012 Peter Colberg
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

#include <halmd/utility/gpu/thread.cuh>
#include <halmd/mdsim/gpu/box_kernel.cuh>
#include <halmd/mdsim/gpu/position_shift_kernel.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {
namespace position_shift_kernel {

/**
 * Shift all positions by 'delta'
 */
template <
    int dimension
    , typename float_type
    , typename ptr_type
    , typename gpu_vector_type
>
__global__ void shift(
    ptr_type g_position
  , gpu_vector_type* g_image
  , unsigned int nparticle
  , unsigned int size
  , fixed_vector<dsfloat, dimension> delta
  , fixed_vector<float, dimension> box_length
)
{
    typedef fixed_vector<float_type, dimension> vector_type;
    typedef fixed_vector<float, dimension> float_vector_type;

    for (unsigned int i = GTID; i < nparticle; i += GTDIM) {
        vector_type r;
        unsigned int species;

        tie(r, species) <<= g_position[i];

        r += delta;
        float_vector_type crossing =
            box_kernel::reduce_periodic(r, box_length);

        g_position[i] <<= tie(r, species);

        if (!(crossing == float_vector_type(0))) {
            g_image[i] =
                crossing + static_cast<float_vector_type>(g_image[i]);
        }
    }
}

/**
 * Shift positions of particle group by 'delta'
 */
template <
    int dimension
    , typename float_type
    , typename ptr_type
    , typename gpu_vector_type
>
__global__ void shift_group(
    ptr_type g_position
  , gpu_vector_type* g_image
  , unsigned int const* g_group
  , unsigned int nparticle
  , unsigned int size
  , fixed_vector<dsfloat, dimension> delta
  , fixed_vector<float, dimension> box_length
)
{
    typedef fixed_vector<float_type, dimension> vector_type;
    typedef fixed_vector<float, dimension> float_vector_type;

    for (unsigned int n = GTID; n < nparticle; n += GTDIM) {
        unsigned int i = g_group[n];

        vector_type r;
        unsigned int species;

        tie(r, species) <<= g_position[i];

        r += delta;
        float_vector_type crossing =
            box_kernel::reduce_periodic(r, box_length);

        g_position[i] <<= tie(r, species);

        if (!(crossing == float_vector_type(0))) {
            g_image[i] =
                crossing + static_cast<float_vector_type>(g_image[i]);
        }
    }
}
} // namespace position_shift_kernel

template <int dimension, typename float_type>
position_shift_wrapper<dimension, float_type>
position_shift_wrapper<dimension, float_type>::kernel = {
    position_shift_kernel::shift<dimension, float_type, ptr_type>
  , position_shift_kernel::shift_group<dimension, float_type, ptr_type>
};

template class position_shift_wrapper<3, float>;
template class position_shift_wrapper<2, float>;

#ifdef USE_GPU_DOUBLE_SINGLE_PRECISION
template class position_shift_wrapper<3, dsfloat>;
template class position_shift_wrapper<2, dsfloat>;
#endif

} // namespace gpu
} // namespace mdsim
} // namespace halmd
