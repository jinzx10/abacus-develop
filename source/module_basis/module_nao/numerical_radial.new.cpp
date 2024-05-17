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
    ngrid_{ngrid},
    cutoff_{space, cutoff},
    f_{new CubicSpline(ngrid, 0, cutoff / (ngrid - 1), value)},
    p_{p},
    sbt_{sbt}
{
    assert(l >= 0 && ngrid > 1 && cutoff > 0.0 && value && p <= 2);

    if (normalize)
    {
        const double dx = cutoff / (ngrid - 1);
        std::vector<double> integrand(value, value + ngrid_);
        for (int i = 0; i < ngrid_; ++i)
        {
            integrand[i] *= integrand[i] * std::pow(i * dx, 2 - p);
        }

        double fac = Integral::simpson(ngrid_, integrand.data(), dx);
        f_->scale(1.0 / std::sqrt(fac));
    }
}


NumericalRadial::NumericalRadial(const NumericalRadial& rhs):
    l_{rhs.l_},
    ngrid_{rhs.ngrid_},
    cutoff_{rhs.cutoff_},
    sbt_{rhs.sbt_}
{
    p_ = rhs.p_;
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
    assert(ngrid > 1 && grid_max >= cutoff_.second);

    double dx = grid_max / (ngrid - 1);

    std::vector<double> grid_new(ngrid);
    std::fill(grid_new.begin(), grid_new.end(),
        [&grid_new, dx](double& x) { return (&x - grid_new.data()) * dx; });

    // NOTE: CubicSpline cannot extrapolate; we need to find the number of
    // new grid points within the cutoff radius before interpolation.
    std::vector<double> value_new(ngrid);
    int ngrid_interp = static_cast<int>(cutoff_.second / dx) + 1;
    f_->eval(ngrid_interp, grid_new.data(), value_new.data());
    f_.reset(new CubicSpline(ngrid, 0, dx, value_new.data()));

    if (g_)
    {
        radrfft();
    }
}


void NumericalRadial::radrfft()
{
    std::vector<double> buffer(3 * ngrid_);
    double* grid_in = buffer.data();
    double* in = grid_in + ngrid_;
    double* out = in + ngrid_;

    const double dx = cutoff_.second / (ngrid_ - 1);
    std::for_each(grid_in, grid_in + ngrid_,
        [dx, &grid_in](double& x) { x = (&x - grid_in) * dx; });

    _eval(cutoff_.first, ngrid_, grid_in, in);

    sbt_.radrfft(l_, ngrid_, f_->xmax(), in, out, p_);
    g_.reset(new CubicSpline(ngrid_, 0, std::acos(-1.0)/dx, out));
}


void NumericalRadial::_eval(
    const Space space,
    const int n,
    const double* const grid,
    double* const value
) const
{
    const auto& f = (space == cutoff_.first) ? f_ : g_;

    assert(n > 0 && grid && value && f);
    assert(std::all_of(grid, grid + n, [](double x) { return x >= 0.0; }));

    std::fill(value, value + n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        if (grid[i] <= f->xmax())
        {
            f->eval(1, &grid[i], &value[i]);
        }
        // values for grid points outside the cutoff radius are zero
    }
}
