###############################################################################
# This file is part of the ANARCHY paper.
# Copyright (c) 2016 Matthieu Schaller (matthieu.schaller@durham.ac.uk)
#               2019 Josh Borrow (joshua.boorrow@durham.ac.uk)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

from numpy import arange, zeros, log, ones


def analytic(time, R_max=0.8, N=200, P0=0.0, rho0=1.0, gas_gamma=5.0 / 3.0):
    solution_r = arange(0, R_max, R_max / N)
    solution_P = zeros(N)
    solution_v_phi = zeros(N)
    solution_v_r = zeros(N)

    for i in range(N):
        if solution_r[i] < 0.2:
            solution_P[i] = P0 + 5.0 + 12.5 * solution_r[i] ** 2
            solution_v_phi[i] = 5.0 * solution_r[i]
        elif solution_r[i] < 0.4:
            solution_P[i] = (
                P0
                + 9.0
                + 12.5 * solution_r[i] ** 2
                - 20.0 * solution_r[i]
                + 4.0 * log(solution_r[i] / 0.2)
            )
            solution_v_phi[i] = 2.0 - 5.0 * solution_r[i]
        else:
            solution_P[i] = P0 + 3.0 + 4.0 * log(2.0)
            solution_v_phi[i] = 0.0

    solution_rho = ones(N) * rho0
    solution_s = solution_P / solution_rho ** gas_gamma
    solution_u = solution_P / ((gas_gamma - 1.0) * solution_rho)

    return dict(
        x=solution_r,
        P=solution_P,
        v_phi=solution_v_phi,
        v_r=solution_v_r,
        u=solution_u,
        rho=solution_rho,
        S=solution_s,
    )
