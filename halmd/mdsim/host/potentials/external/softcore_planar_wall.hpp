/*
 * Copyright © 2026      Tolga Atalay
 * Copyright © 2014-2015 Sutapa Roy
 * Copyright © 2014-2015 Felix Höfling
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

#ifndef HALMD_MDSIM_HOST_POTENTIALS_EXTERNAL_SOFTCORE_PLANAR_WALL_HPP
#define HALMD_MDSIM_HOST_POTENTIALS_EXTERNAL_SOFTCORE_PLANAR_WALL_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <cmath>
#include <lua.hpp>
#include <memory>
#include <tuple>

#include <halmd/io/logger.hpp>
#include <halmd/numeric/blas/fixed_vector.hpp>
#include <halmd/utility/signal.hpp>

namespace halmd {
namespace mdsim {
namespace host {
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

    /**
     * Compute force and potential energy due to soft-core planar walls.
     * Soft-core WCA planar wall.
     *
     * The signed distance from the wall is
     *
     *   d = dot(r, n) - offset.
     *
     * The surface normal points out of the accessible region. The wall acts
     * only on particles with d < 0. Their distance from the wall and the
     * lambda-dependent effective distance are
     *
     *   distance = -d,
     *   r_soft = distance + (1 - lambda) * z_cw.
     *
     * and the potential is
     *
     *   u(lambda, d) =
     *       4 * epsilon * lambda^2
     *       * [(sigma / r_soft)^12
     *          - (sigma / r_soft)^6
     *          + 1/4]
     *       * w(distance).
     *
     * For d >= 0, both force and potential energy are zero.
     *
     */
    std::tuple<vector_type, float_type> operator()(vector_type const& r, unsigned int species) const
    {
        float_type en_pot = 0;
        vector_type force = 0;

        // loop over walls
        for (unsigned int i = 0; i < surface_normal_.size(); ++i) {

          // Signed distance from particle to wall plane
          float_type d = inner_prod(r, surface_normal_(i)) - offset_(i);

          // The normal points out of the accessible region. Apply the wall
          // only on the inner side of the plane.
          if (d >= 0)
              continue;

          float_type distance = -d;

          float_type z_cw = cutoff_(i, species);

          // truncate interaction
          if (distance >= z_cw)
              continue;

          float_type epsilon = epsilon_(i, species);
          float_type sigma = sigma_(i, species);
          float_type lambda = lambda_;
          float_type lambda2 = lambda * lambda;
          float_type h = smoothing_;

          float_type soft_distance = distance + (1 - lambda) * z_cw;

          float_type x = sigma / soft_distance;
          float_type x6 = std::pow(x, 6);
          float_type x12 = x6 * x6;

          float_type bracket = x12 - x6 + 0.25;

          float_type delta = distance - z_cw;
          float_type delta3 = std::pow(delta, 3);
          float_type delta4 = delta3 * delta;

          float_type h4 = std::pow(h, 4);

          float_type switch_denominator = 1 + h4 * delta4;

          float_type cutoff_switch = delta4 / switch_denominator;

          float_type d_switch_ddistance = 4 * delta3
              / (switch_denominator * switch_denominator);

          float_type energy_raw = 4 * epsilon * lambda2 * bracket;

          float_type en_wall = energy_raw * cutoff_switch;

          float_type force_core =
              24 * epsilon * lambda2
                 * (2 * x12 - x6)
                 / soft_distance;
          float_type const force_magnitude =
              force_core * cutoff_switch
              - energy_raw * d_switch_ddistance;

          // accumulate force and potential energy
          force -= force_magnitude * surface_normal_(i);
          en_pot += en_wall;
       }


        return std::make_tuple(force, en_pot);
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

    /** module logger */
    std::shared_ptr<logger> logger_;

    /** wall offset update signal */
    signal_type on_set_offset_;
};

} // namespace external
} // namespace potentials
} // namespace host
} // namespace mdsim
} // namespace halmd

#endif /* ! HALMD_MDSIM_HOST_POTENTIALS_EXTERNAL_SOFTCORE_PLANAR_WALL_HPP */
