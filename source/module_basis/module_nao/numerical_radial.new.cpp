#include "module_basis/module_nao/numerical_radial.new.h"

#include <cmath>
#include <algorithm>
#include <cassert>

#include "module_base/math_integral.h"

using ModuleBase::CubicSpline;
using ModuleBase::Integral;

NumericalRadial::NumericalRadial(
    const int l,
    const int ngrid,
    const double cutoff,
    const double* const value,
    const Space space,
    const bool normalize,
    const int p,
    const ModuleBase::SphericalBesselTransformer& sbt
):
    l_{l},
    cutoff_space_{space},
    cutoff_radius_{cutoff},
    f_{new CubicSpline(ngrid, 0, cutoff / (ngrid - 1), value)},
    p_{p},
    sbt_{sbt}
{
    assert(l >= 0 && ngrid > 1 && cutoff > 0.0 && value && p <= 2);

    if (normalize)
    {
        const double dx = cutoff / (ngrid - 1);
        std::vector<double> integrand(value, value + ngrid);
        for (int i = 0; i < ngrid; ++i)
        {
            integrand[i] *= integrand[i] * std::pow(i * dx, 2 - p);
        }

        const double fac = Integral::simpson(ngrid, integrand.data(), dx);
        f_->scale(1.0 / std::sqrt(fac));
    }
}


NumericalRadial::NumericalRadial(const NumericalRadial& rhs):
    l_{rhs.l_},
    cutoff_space_{rhs.cutoff_space_},
    cutoff_radius_{rhs.cutoff_radius_},
    p_{rhs.p_},
    sbt_{rhs.sbt_}
{
    f_.reset(rhs.f_ ? new CubicSpline(*rhs.f_) : nullptr);
    g_.reset(rhs.g_ ? new CubicSpline(*rhs.g_) : nullptr);
}


NumericalRadial& NumericalRadial::operator=(NumericalRadial rhs)
{
    std::swap(*this, rhs);
    return *this;
}


void NumericalRadial::rvalue(const int n, const double* const grid, double* const value)
{
    _eval(Space::r, n, grid, value);
}


void NumericalRadial::kvalue(const int n, const double* const grid, double* const value)
{
    _eval(Space::k, n, grid, value);
}


void NumericalRadial::set_transformer(const ModuleBase::SphericalBesselTransformer& sbt)
{
    sbt_ = sbt;
}


void NumericalRadial::set_grid(const int ngrid, const double grid_max)
{
    assert(ngrid > 1 && grid_max >= cutoff_radius_);

    const double dx = grid_max / (ngrid - 1);
    std::vector<double> grid_new(ngrid);
    std::fill(grid_new.begin(), grid_new.end(),
        [&grid_new, dx](double& x) { return (&x - grid_new.data()) * dx; });

    // NOTE: CubicSpline cannot extrapolate; we need to find the number of
    // new grid points within the cutoff radius before interpolation.
    std::vector<double> value_new(ngrid, 0.0);
    const int ngrid_interp = static_cast<int>(cutoff_radius_ / dx) + 1;
    f_->eval(ngrid_interp, grid_new.data(), value_new.data());
    f_.reset(new CubicSpline(ngrid, 0, dx, value_new.data()));

    if (g_)
    {
        radrfft();
    }
}


void NumericalRadial::radrfft()
{
    const int ngrid = f_->n();
    std::vector<double> out(ngrid);
    const double dx = cutoff_radius_ / (ngrid - 1);

    sbt_.radrfft(l_, ngrid, f_->xmax(), f_->ydata(), out.data(), p_);
    g_.reset(new CubicSpline(ngrid, 0, std::acos(-1.0)/dx, out.data()));
}


void NumericalRadial::_eval(
    const Space space,
    const int n,
    const double* const grid,
    double* const value
) const
{
    const auto& f = (space == cutoff_space_) ? f_ : g_;

    assert(n > 0 && grid && value && f);
    assert(std::all_of(grid, grid + n, [](double x) { return x >= 0.0; }));

    std::transform(grid, grid + n, value, [&f](const double& x)
    {
        double val = 0.0;
        return x <= f->xmax() ? (f->eval(1, &x, &val), val): 0.0;
    });
}


