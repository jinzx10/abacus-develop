#ifndef NUMERICAL_RADIAL_H_
#define NUMERICAL_RADIAL_H_

#include <memory>
#include <utility>

#include "module_base/cubic_spline.h"
#include "module_base/spherical_bessel_transformer.h"

/**
 * @brief Numerical radial function.
 *
 * This container is supposed to hold the radial part of a pseudo-atomic orbital
 * ([radial] x [spherical harmonic]) which has some cutoff radius in some space,
 * like numerical atomic orbitals, Kleinman-Bylander beta functions, etc. Such
 * a radial function is associated with some angular momentum l, and its values
 * in r & k space are related by an l-th order spherical Bessel transform.
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
     * @brief Initializes the object in the space with strict cutoff.
     *
     * This constructor initializes the object with the given angular momentum,
     * number of grid points, cutoff radius, and values on the grid. The grid is
     * assumed to be uniform:
     *
     *                    cutoff
     *      grid[i] = i * -------
     *                    ngrid-1
     *
     * and the value may carry an extra exponent p than what the object is supposed
     * to represent (see @ref p_ for details).
     *
     * @param[in]   l           angular momentum
     * @param[in]   ngrid       number of grid points
     * @param[in]   cutoff      cutoff radius
     * @param[in]   value       values on the grid
     * @param[in]   space       the space (r or k) of inputs
     * @param[in]   normalize   whether to normalize the radial function
     * @param[in]   p           extra exponent in input values
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
     * @brief Performs a FFT-based spherical Bessel transform from the space with
     * strict cutoff to the other.
     *
     */
    void radrfft();


    /** 
     * @brief Sets a SphericalBesselTransformer.
     *
     * Spherical Bessel transforms (radrfft) of this class are performed via
     * SphericalBesselTransformer objects (sbt_), which are opaque shared pointers.
     * By default each NumericalRadial object constructs its own sbt_ in cache-
     * disabled mode, so calls to radrfft on different objects are independent.
     *
     * This function enables an object to switch to a designated sbt. Once multiple
     * NumericalRadial objects of the same grid are supplied with a common cache-
     * enable sbt, their radrfft will benefit from the cache mechanism.
     *
     */
    void set_transformer(const ModuleBase::SphericalBesselTransformer& sbt);


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
    void set_grid(const int ngrid,const double grid_max);


    int l() const { return l_; }
    int ngrid() const { return ngrid_; }

    void rvalue(const int n, const double* const grid, double* const value);
    void kvalue(const int n, const double* const grid, double* const value);

    int pr() const { return cutoff_.first == Space::r ? p_ : 0; }
    int pk() const { return cutoff_.first == Space::k ? p_ : 0; }

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
     * than f(r). For example, Kleinman-Bylander beta functions in UPF files are
     * often provided as r*beta(r) instead of bare beta(r). Very often one merely
     * needs to perform multiplications upon r*beta(r); there's no need to recover
     * the bare beta(r).
     *
     * This class takes care of this kind of situations. For example, when constructing
     * a beta function's numerial radial object, one can simply feed r*beta(r) to value
     * and specify p = 1 so that the class would take this extra exponent into account
     * when performing spherical Bessel transforms.
     *
     * @note Internally, values stored in CubicSpline objects still carry this exponent,
     * hence the outputs of rvalue() or kvalue(). One should be aware of this when using
     * these values from outside and remember to get the extra exponents via pr() or pk().
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
