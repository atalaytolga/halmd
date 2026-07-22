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

#include <boost/numeric/ublas/io.hpp>
#include <cuda_wrapper/cuda_wrapper.hpp>
#include <stdexcept>
#include <string>

#include <halmd/mdsim/gpu/forces/external.hpp>
#include <halmd/mdsim/gpu/potentials/external/softcore_planar_wall.hpp>
#include <halmd/mdsim/gpu/potentials/external/softcore_planar_wall_kernel.hpp>
#include <halmd/utility/lua/lua.hpp>

using namespace std;

namespace halmd {
namespace mdsim {
namespace gpu {
namespace potentials {
namespace external {

/**
 * Initialise softcore_planar_wall potential
 */
template <int dimension, typename float_type>
softcore_planar_wall<dimension, float_type>::softcore_planar_wall(
    scalar_container_type const& offset
  , vector_container_type const& surface_normal
  , matrix_container_type const& epsilon
  , matrix_container_type const& sigma
  , matrix_container_type const& cutoff
  , float_type smoothing
  , float_type lambda
  , shared_ptr<logger> logger
)
  // initialise attributes
  : offset_(offset)
  , surface_normal_(surface_normal)
  , epsilon_(epsilon)
  , sigma_(sigma)
  , cutoff_(cutoff)
  , smoothing_(smoothing)
  , lambda_(lambda)
  , g_param_geometry_(surface_normal_.size())
  , g_param_potential_(epsilon_.size1() * epsilon_.size2())
  , t_param_geometry_(g_param_geometry_)
  , t_param_potential_(g_param_potential_)
  , logger_(logger)
{
    unsigned int const nwall = epsilon_.size1();
    unsigned int const nspecies = epsilon_.size2();

    // check parameter size
    if (offset_.size() != surface_normal_.size()
     || offset_.size() != epsilon_.size1()
     || offset_.size() != sigma_.size1()
     || offset_.size() != cutoff_.size1()
       ) {
        throw invalid_argument("geometry parameters have mismatching shapes");
    }

    if (epsilon_.size2() != sigma_.size2()
     || epsilon_.size2() != cutoff_.size2()
       ) {
        throw invalid_argument("potential parameters have mismatching shapes");
    }

    if (!(lambda_ >= 0 && lambda_ <= 1)) {
        throw invalid_argument("coupling parameter lambda must be in the interval [0, 1]");
    }

    LOG("number of walls: " << nwall);
    LOG("wall position: d₀ = " << offset_);
    LOG("surface normal: n = " << surface_normal_);
    LOG("interaction strength: epsilon = " << epsilon_);
    LOG("interaction range: sigma = " << sigma_);
    LOG("coupling parameter: lambda = " << lambda_);
    LOG("cutoff distance: rc = " << cutoff_);
    LOG("smoothing parameter: h = " << smoothing_);

    // impose normalisation of surface normals
    for (auto& n : surface_normal_) {
        float_type const norm = norm_2(n);
        if (!(norm > 0)) {
            throw invalid_argument("surface normal must be non-zero");
        }
        n /= norm;
    }

    // merge geometry parameters in a single array and copy to device
    cuda::memory::host::vector<float4> param_geometry(g_param_geometry_.size());
    for (size_t i = 0; i < nwall; ++i) {
        param_geometry[i] <<= tie(surface_normal_(i), offset_(i));
    }
    cuda::copy(param_geometry.begin(), param_geometry.end(), g_param_geometry_.begin());

    // merge potential parameters in a single array and copy to device
    cuda::memory::host::vector<float4> param_potential(g_param_potential_.size());
    for (size_t i = 0; i < nwall; ++i) {
        for (size_t j = 0; j < nspecies; ++j) {
            using namespace softcore_planar_wall_kernel;

            fixed_vector<float, 4> p = 0;
            p[EPSILON] = epsilon_(i, j);
            p[SIGMA]   = sigma_(i, j);
            p[CUTOFF]  = cutoff_(i, j);
            param_potential[j * nwall + i] = p;
        }
    }
    cuda::copy(param_potential.begin(), param_potential.end(), g_param_potential_.begin());
}

template <int dimension, typename float_type>
void softcore_planar_wall<dimension, float_type>::set_offset(
    scalar_container_type const& offset
)
{
    size_t const nwall = offset_.size();

    if (offset.size() != nwall) {
        throw invalid_argument(
            "number of wall offsets does not match number of walls"
        );
    }

    offset_ = offset;

    cuda::memory::host::vector<float4> param_geometry(
        g_param_geometry_.size()
    );
    for (size_t i = 0; i < nwall; ++i) {
        param_geometry[i] <<= tie(surface_normal_(i), offset_(i));
    }
    cuda::copy(
        param_geometry.begin(),
        param_geometry.end(),
        g_param_geometry_.begin()
    );
    on_set_offset_();
}

template <int dimension, typename float_type>
void softcore_planar_wall<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luaponte;
    static string class_name("softcore_planar_wall_" + to_string(dimension));
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("gpu")
            [
                namespace_("potentials")
                [
                    namespace_("external")
                    [
                        class_<softcore_planar_wall, shared_ptr<softcore_planar_wall>>(class_name.c_str())
                            .def(constructor<
                                 scalar_container_type const&
                               , vector_container_type const&
                               , matrix_container_type const&
                               , matrix_container_type const&
                               , matrix_container_type const&
                               , float_type
                               , float_type
                               , shared_ptr<logger>
                             >())
                            .def("set_offset", &softcore_planar_wall::set_offset)
                            .def("on_set_offset", &softcore_planar_wall::on_set_offset)
                            .property("offset", &softcore_planar_wall::offset)
                            .property("surface_normal", &softcore_planar_wall::surface_normal)
                            .property("epsilon", &softcore_planar_wall::epsilon)
                            .property("sigma", &softcore_planar_wall::sigma)
                            .property("lambda", &softcore_planar_wall::lambda)
                            .property("cutoff", &softcore_planar_wall::cutoff)
                            .property("smoothing", &softcore_planar_wall::smoothing)
                    ]
                ]
            ]
        ]
    ];
}


HALMD_LUA_API int luaopen_libhalmd_mdsim_gpu_potentials_external_softcore_planar_wall(lua_State* L)
{
    softcore_planar_wall<3, float>::luaopen(L);
    softcore_planar_wall<2, float>::luaopen(L);
#ifdef USE_GPU_SINGLE_PRECISION
    forces::external<3, float, softcore_planar_wall<3, float>>::luaopen(L);
    forces::external<2, float, softcore_planar_wall<2, float>>::luaopen(L);
#endif
#ifdef USE_GPU_DOUBLE_SINGLE_PRECISION
    forces::external<3, dsfloat, softcore_planar_wall<3, float>>::luaopen(L);
    forces::external<2, dsfloat, softcore_planar_wall<2, float>>::luaopen(L);
#endif
    return 0;
}

// explicit instantiation
template class softcore_planar_wall<3, float>;
template class softcore_planar_wall<2, float>;

} // namespace external
} // namespace potentials

namespace forces {

// explicit instantiation of force modules
using namespace potentials::external;

#ifdef USE_GPU_SINGLE_PRECISION
template class external<3, float, softcore_planar_wall<3, float>>;
template class external<2, float, softcore_planar_wall<2, float>>;
#endif
#ifdef USE_GPU_DOUBLE_SINGLE_PRECISION
template class external<3, dsfloat, softcore_planar_wall<3, float>>;
template class external<2, dsfloat, softcore_planar_wall<2, float>>;
#endif

} // namespace forces
} // namespace gpu
} // namespace mdsim
} // namespace halmd
