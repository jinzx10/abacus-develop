#ifndef ISTATE_ENVELOPE_H
#define ISTATE_ENVELOPE_H
#include "src_lcao/local_orbital_wfc.h"
#include "src_lcao/gint_gamma.h"
#include "src_lcao/gint_k.h"
#include "src_pw/pw_basis.h"
#include "module_psi/psi.h"

class IState_Envelope
{
public:
    IState_Envelope();
    ~IState_Envelope();

    /// for gamma_only
    void begin(const psi::Psi<double>* psid, Local_Orbital_wfc& lowf, Gint_Gamma& gg, int& out_wfc_pw, int& out_wfc_r);
    /// for multi-k
    void begin(const psi::Psi<std::complex<double>>* psi, Local_Orbital_wfc& lowf, Gint_k& gk, int& out_wfc_pw, int& out_wfc_r);

private:
    bool* bands_picked;

    void set_pw_wfc(PW_Basis& pwb,
        const int& ik, const int& ib,
        const int& nspin,
        const double* const* const rho,
        psi::Psi<std::complex<double>> &wfc_g);

};
#endif
