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

#include <halmd/config.hpp>

#define BOOST_TEST_MODULE softcore_planar_wall
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

#include <halmd/mdsim/host/potentials/external/softcore_planar_wall.hpp>
#include <halmd/utility/lua/lua.hpp>
#include <test/tools/ctest.hpp>
#include <test/tools/lua.hpp>
#ifdef HALMD_WITH_GPU
# include <halmd/mdsim/gpu/potentials/external/softcore_planar_wall.hpp>
# include <test/tools/cuda.hpp>
#endif

using namespace halmd;

#ifndef USE_HOST_SINGLE_PRECISION
using host_float_type = double;
#else
using host_float_type = float;
#endif

template <typename potential_type>
std::shared_ptr<potential_type> make_potential(double lambda)
{
    using vector_type = typename potential_type::vector_type;
    typename potential_type::scalar_container_type offset(2, 1);
    typename potential_type::vector_container_type normal(2, vector_type(0));
    normal(0)[0] = 2; // constructor must normalise the normal
    normal(1)[0] = -3;
    typename potential_type::matrix_container_type epsilon(2, 2), sigma(2, 2), cutoff(2, 2);
    for (unsigned int i = 0; i < 2; ++i) {
        for (unsigned int s = 0; s < 2; ++s) {
            epsilon(i, s) = 0.5 + i + s;
            sigma(i, s) = 0.7 + 0.1 * i + 0.2 * s;
            cutoff(i, s) = 1.5 + 0.2 * i + 0.3 * s;
        }
    }
    return std::make_shared<potential_type>(offset, normal, epsilon, sigma, cutoff, 0.1, lambda);
}

template <int dimension, typename potential_type>
void check_derivative()
{
    using reference_type = mdsim::host::potentials::external::softcore_planar_wall<dimension, host_float_type>;
    // Account for both the reference energy precision and the queried potential precision.
    bool const single_precision = sizeof(host_float_type) == sizeof(float)
        || sizeof(typename potential_type::vector_type::value_type) == sizeof(float);
    double const step = single_precision ? 0.0002 : 0.000001;
    double const tolerance = single_precision ? 0.01 : 0.000001;
    for (double lambda : {0., 0.25, 0.7, 1.}) {
        auto potential = make_potential<potential_type>(lambda);
        auto energy = [&](double coupling, double x, unsigned int species) {
            auto reference = make_potential<reference_type>(coupling);
            typename reference_type::vector_type r(0);
            r[0] = x;
            return static_cast<double>(std::get<1>((*reference)(r, species)));
        };
        for (unsigned int species = 0; species < 2; ++species) {
            for (double x : {-3., -1.5, -1., -0.5, -0.3, 0., 0.8, 1., 1.5, 3.}) {
                typename potential_type::vector_type r(0);
                r[0] = x;
                double finite_difference;
                if (lambda == 0) {
                    finite_difference = (-3 * energy(0, x, species)
                        + 4 * energy(step, x, species) - energy(2 * step, x, species)) / (2 * step);
                } else if (lambda == 1) {
                    finite_difference = (3 * energy(1, x, species)
                        - 4 * energy(1 - step, x, species) + energy(1 - 2 * step, x, species)) / (2 * step);
                } else {
                    finite_difference = (energy(lambda + step, x, species)
                        - energy(lambda - step, x, species)) / (2 * step);
                }
                double const derivative = potential->du_dlambda(r, species);
                BOOST_CHECK_SMALL(derivative - finite_difference,
                    tolerance * std::max(1., std::abs(finite_difference)));
                if (lambda == 0 || std::abs(x) == 3) {
                    BOOST_CHECK_EQUAL(derivative, 0);
                }
            }
        }
        typename potential_type::vector_type r(0);
        BOOST_CHECK_THROW(potential->du_dlambda(r, 2), std::invalid_argument);
        BOOST_CHECK_THROW(potential->du_dlambda(r, std::numeric_limits<unsigned int>::max()), std::invalid_argument);

        // Offset updates must immediately affect the point query.
        typename potential_type::scalar_container_type offset(2, 5);
        potential->set_offset(offset);
        BOOST_CHECK_EQUAL(potential->du_dlambda(r, 0), 0);
    }
}

template <typename potential_type>
void check_exclusions()
{
    using vector_type = typename potential_type::vector_type;
    typename potential_type::scalar_container_type offset(1, 0);
    typename potential_type::vector_container_type normal(1, vector_type(0));
    normal(0)[0] = 1;
    typename potential_type::matrix_container_type epsilon(1, 1, 1);
    typename potential_type::matrix_container_type sigma(1, 1, 1);
    typename potential_type::matrix_container_type cutoff(1, 1, 2);
    // With no smoothing, evaluating the switch at the cutoff would give 0/0.
    potential_type potential(offset, normal, epsilon, sigma, cutoff, 0, 1);
    for (double x : {-3., -2., 0., 1.}) {
        vector_type r(0);
        r[0] = x;
        BOOST_CHECK_EQUAL(potential.du_dlambda(r, 0), 0);
    }
    epsilon(0, 0) = 0;
    potential_type disabled(offset, normal, epsilon, sigma, cutoff, 0.1, 0.7);
    vector_type r(0);
    r[0] = -0.5;
    BOOST_CHECK_EQUAL(disabled.du_dlambda(r, 0), 0);
}

template <int dimension, typename potential_type>
void check_lua()
{
    lua_test_fixture fixture;
    potential_type::luaopen(fixture.L);
    auto potential = make_potential<potential_type>(0.7);
    typename potential_type::vector_type r(0);
    luaponte::globals(fixture.L)["potential"] = potential;
    luaponte::globals(fixture.L)["dimension"] = dimension;
    luaponte::globals(fixture.L)["expected"] = potential->du_dlambda(r, 1);
    BOOST_REQUIRE_MESSAGE(fixture.dostring(
        "position = {}; for i = 1, dimension do position[i] = 0 end\n"
        "assert(math.abs(potential:du_dlambda(position, 1) - expected) < 1e-6)\n"
        "assert(not pcall(function() potential:du_dlambda(position, 2) end))\n"
        "assert(not pcall(function() potential:du_dlambda({}, 0) end))\n"
        "potential:set_offset({5, 5})\n"
        "assert(potential:du_dlambda(position, 0) == 0)\n"
    ), lua_test_fixture::error(fixture.L));
}

BOOST_AUTO_TEST_CASE(softcore_planar_wall_host)
{
    using potential2 = mdsim::host::potentials::external::softcore_planar_wall<2, host_float_type>;
    using potential3 = mdsim::host::potentials::external::softcore_planar_wall<3, host_float_type>;
    check_derivative<2, potential2>();
    check_derivative<3, potential3>();
    check_exclusions<potential2>();
    check_exclusions<potential3>();
    check_lua<2, potential2>();
    check_lua<3, potential3>();
}

#ifdef HALMD_WITH_GPU
BOOST_FIXTURE_TEST_CASE(softcore_planar_wall_gpu, set_cuda_device)
{
    using potential2 = mdsim::gpu::potentials::external::softcore_planar_wall<2, float>;
    using potential3 = mdsim::gpu::potentials::external::softcore_planar_wall<3, float>;
    check_derivative<2, potential2>();
    check_derivative<3, potential3>();
    check_exclusions<potential2>();
    check_exclusions<potential3>();
    check_lua<2, potential2>();
    check_lua<3, potential3>();
}
#endif
