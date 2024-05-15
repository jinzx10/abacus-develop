#ifndef NUMERICAL_RADIAL_H_
#define NUMERICAL_RADIAL_H_

#include <memory>
#include <utility>

#include "module_base/cubic_spline.h"
#include "module_base/spherical_bessel_transformer.h"

/**
 * @brief Numerical radial function.
 *
 * This class is designed to be the container for the radial part of
 * numerical atomic orbitals, Kleinman-Bylander beta functions, and all
 * other similar numerical radial functions in three-dimensional space,
 * each of which is associated with some angular momentum l and whose r
 * and k space values are related by an l-th order spherical Bessel transform.
 *
 * A NumericalRadial object can be initialized by "build", which requires
 * the angular momentum, the number of grid points, the grid and the
 * corresponding values. Grid does not have to be uniform. One can initialize
 * the object in either r or k space. After initialization, one can set the
 * grid in the other space via set_grid or set_uniform_grid. Values in the
 * other space are automatically computed by a spherical Bessel transform.
 *
 * Usage:
 *
 *      // Prepares the grid & values to initialize the objects
 *      int sz = 2000;
 *      double dr = 0.01;
 *      double* grid = new double[sz];
 *      for (int ir = 0; ir != sz; ++ir) {
 *          grid[ir] = ir * dr; 
 *          f[ir] = std::exp(-grid[ir] * grid[ir]);
 *      }
 *      // grid does not necessarily have to be uniform; it just
 *      // has to be positive and strictly increasing.
 *
 *      // The class will interpret the input values as r^p * F(r)
 *      // where F is the underlying radial function that the class object
 *      // actually represents.
 *      int p1 = 0;
 *      int p2 = -2;
 *
 *       
 *      NumericalRadial chi1();
 *      NumericalRadial chi2;
 *      chi1.build(0, true, sz, grid, f, p1);
 *      chi2.build(2, true, sz, grid, f, p2);
 *
 *      // Now chi1 represents exp(-r^2); chi2 actually represents
 *      // r^2*exp(-r^2), even though the values stored is also exp(-r^2).
 *
 *      // Adds the k-space grid.
 *      chi1.set_uniform_grid(false, sz, PI/dr, 't');
 *      chi2.set_uniform_grid(false, sz, PI/dr, 't');
 *      // k-space values are automatically computed above
 *
 */
class NumericalRadial
{

public:

    enum class Space { r, k };

    NumericalRadial() = delete;
    ~NumericalRadial() = default;

    NumericalRadial(const NumericalRadial&);
    NumericalRadial(NumericalRadial&&) = default;

    NumericalRadial& operator=(NumericalRadial); // copy-swap idiom

    /**
     * @brief Initializes the object by providing the grid & values in one space.
     *
     * @param[in]   l           angular momentum
     * @param[in]   ngrid       number of grid points
     * @param[in]   cutoff      cutoff radius
     * @param[in]   value       values on the grid
     * @param[in]   space       specifies whether the input grid & values are r or k space
     * @param[in]   normalize   whether to normalize the radial function
     * @param[in]   p           implicit exponent in input values (see @ref pr_ & @ref pk_)
     * @param[in]   sbt         a user-provided spherical Bessel transformer
     *
     */
    NumericalRadial(
        const int l,
        const int ngrid,
        const double cutoff,
        const double* const value,
        const Space space = Space::r,
        const bool normalize = false,
        const int p = 0,
        const ModuleBase::SphericalBesselTransformer& sbt = {}
    );


    /** 
     * @brief Sets a SphericalBesselTransformer.
     * 
     * By default the class uses its own SphericalBesselTransformer in cache-disabled
     * mode. Alternatively, one may set up a SphericalBesselTransformer in cache-enabled
     * mode and have it shared among multiple NumericalRadial objects by calling this
     * function, which could be beneficial when there are a lot of NumericalRadial objects
     * of the same grid.
     *
     */
    void set_transformer(
        const ModuleBase::SphericalBesselTransformer& sbt
    );


    /**
     * @brief Resets the grid (and update values) in the space with strict cutoff.
     *
     * This function resets the grid in the space with strict cutoff (i.e., the space
     * specified in constructor) and updates values by an interpolation.
     *
     * If values in the other space also exist, they will be updated by a spherical
     * Bessel transform (radrfft).
     *
     * @note it is not allowed to set a cutoff smaller than the actual cutoff_.
     *
     */
    void set_grid(
        const int ngrid,
        const double grid_max
    );


    /**
     * @brief Performs a FFT-based spherical Bessel transform from the space with strict
     * cutoff to the other space.
     *
     */
    void radrfft();


    int l() const { return l_; }
    int ngrid() const { return ngrid_; }

    void rvalue(const int n, const double* const grid, double* const value);
    void kvalue(const int n, const double* const grid, double* const value);

    double pr() const { return cutoff_.first == Space::r ? p_ : 0; }
    double pk() const { return cutoff_.first == Space::k ? p_ : 0; }

    // cutoff space & radius
    std::pair<Space, double> cutoff() const { return cutoff_; }


private:

    /// angular momentum
    int l_;

    /// number of grid points
    int ngrid_;

    /**
     * @brief Cutoff space and radius.
     *
     * A radial function with a strict cutoff in r space cannot have a strict
     * cutoff in k space (and vice versa), hence only one cutoff is recorded,
     * which includes the space and the radius.
     * 
     * Note that the grid in the space with a strict cutoff might be reset via
     * set_grid with a larger "cutoff" than the one provided in the constructor
     * for the sake of FFT-based spherical Bessel transform and two-center table.
     * The following variable keeps track of the original cutoff radius, i.e.,
     * the one given in the constructor.
     *
     */
    std::pair<Space, double> cutoff_;

    /// radial function in the space with strict cutoff
    std::unique_ptr<ModuleBase::CubicSpline> f_;

    /// radial function in the space without strict cutoff
    std::unique_ptr<ModuleBase::CubicSpline> g_;

    /**
     * @name Extra exponent in input values.
     *
     * Sometimes a radial function is given in the form of pow(r,p) * f(r) rather
     * than f(r). For example, the Kleinman-Bylander beta functions in a UPF file
     * are often given as r*beta(r) instead of bare beta(r). Very often r*beta(r)
     * is adequate for all practical purposes; there's no need to recover the bare
     * beta(r).
     *
     * This class takes care of this situation. When building the object, one can
     * simply feed pow(r,p) * f(r) to value and specify the exponent p so that the
     * class would know that the values have an extra exponent. This variable keeps
     * track of this exponent, which will be automatically taken account during
     * spherical Bessel transforms.
     *
     */
    int p_ = 0;

    /// An object that provides spherical Bessel transforms (possibly with cache)
    ModuleBase::SphericalBesselTransformer sbt_;


    /// Evaluates the radial function on some grid.
    void _eval(
        const Space space,
        const int ngrid,
        const double* const grid,
        double* const value
    ) const;

};

#endif
