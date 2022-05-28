#include "ELEC_scf.h"
#include "../src_pw/global.h"
#include "../src_io/chi0_hilbert.h"
#include "../src_io/print_info.h"
#include "../src_pw/symmetry_rho.h"
#include "dftu.h"
#include "LCAO_evolve.h"
#include "ELEC_cbands_k.h"
#include "ELEC_cbands_gamma.h"
#include "ELEC_evolve.h"
#include "input_update.h"
#include "../src_pw/occupy.h"
#include "../module_base/timer.h"
#include "chrono"
//new
#include "../src_pw/H_Ewald_pw.h"
#ifdef __DEEPKS
#include "../module_deepks/LCAO_deepks.h"	//caoyu add 2021-06-04
#endif

ELEC_scf::ELEC_scf() {}
ELEC_scf::~ELEC_scf() {}

int ELEC_scf::iter = 0;

void ELEC_scf::scf(const int& istep,
    Local_Orbital_Charge& loc,
    Local_Orbital_wfc& lowf,
    LCAO_Hamilt& uhm)
{
    ModuleBase::TITLE("ELEC_scf", "scf");
    ModuleBase::timer::tick("ELEC_scf", "scf");

    // (1) calculate ewald energy.
    // mohan update 2021-02-25
    H_Ewald_pw::compute_ewald(GlobalC::ucell, GlobalC::rhopw);

    // mohan add 2012-02-08
    set_pw_diag_thr();

    // the electron charge density should be symmetrized,
    // here is the initialization
    Symmetry_rho srho;
    for (int is = 0; is < GlobalV::NSPIN; is++)
    {
        srho.begin(is, GlobalC::CHR, GlobalC::pw, GlobalC::Pgrid, GlobalC::symm);
    }

    //	std::cout << scientific;
    //	std::cout << std::setiosflags(ios::fixed);

    if (GlobalV::OUT_LEVEL == "ie" || GlobalV::OUT_LEVEL == "m")
    {
        if (GlobalV::COLOUR && GlobalV::MY_RANK == 0)
        {
            printf("\e[33m%-7s\e[0m", "ITER");
            printf("\e[33m%-15s\e[0m", "ETOT(Ry)");
            if (GlobalV::NSPIN == 2)
            {
                printf("\e[33m%-10s\e[0m", "TMAG");
                printf("\e[33m%-10s\e[0m", "AMAG");
            }
            printf("\e[33m%-14s\e[0m", "SCF_THR");
            printf("\e[33m%-15s\e[0m", "ETOT(eV)");
            printf("\e[33m%-11s\e[0m\n", "TIME(s)");
        }
        else
        {
            std::cout << " " << std::setw(7) << "ITER";

            if (GlobalV::NSPIN == 2)
            {
                std::cout << std::setw(10) << "TMAG";
                std::cout << std::setw(10) << "AMAG";
            }

            std::cout << std::setw(15) << "ETOT(eV)";
            std::cout << std::setw(15) << "EDIFF(eV)";
            std::cout << std::setw(11) << "SCF_THR";
            std::cout << std::setw(11) << "TIME(s)" << std::endl;
        }
    }// end GlobalV::OUT_LEVEL


    for (iter = 1; iter <= GlobalV::SCF_NMAX; iter++)
    {
        Print_Info::print_scf(istep, iter);

        std::string ufile = "CHANGE";
        Update_input UI;
        UI.init(ufile);

        if (INPUT.dft_plus_u) GlobalC::dftu.iter_dftu = iter;
#ifdef __MPI
        auto clock_start = MPI_Wtime();
#else
        auto clock_start = std::chrono::system_clock::now();
#endif
        conv_elec = false;//mohan add 2008-05-25

        // mohan add 2010-07-16
        // used for pulay mixing.
        if (iter == 1)
        {
            GlobalC::CHR.set_new_e_iteration(true);
        }
        else
        {
            GlobalC::CHR.set_new_e_iteration(false);
        }

        // set converged threshold,
        // automatically updated during self consistency, only for CG.
        this->update_pw_diag_thr(iter);
        if (GlobalV::FINAL_SCF && iter == 1)
        {
            init_mixstep_final_scf();
            //GlobalC::CHR.irstep=0;
            //GlobalC::CHR.idstep=0;
            //GlobalC::CHR.totstep=0;
        }

        // mohan update 2012-06-05
        GlobalC::en.calculate_harris(1);

        // mohan move it outside 2011-01-13
        // first need to calculate the weight according to
        // electrons number.
        // mohan add iter > 1 on 2011-04-02
        // because the GlobalC::en.ekb has not value now.
        // so the smearing can not be done.
        if (iter > 1)Occupy::calculate_weights();

        if (GlobalC::wf.init_wfc == "file")
        {
            if (iter == 1)
            {
                std::cout << " WAVEFUN -> CHARGE " << std::endl;

                // The occupation should be read in together.
                // Occupy::calculate_weights(); //mohan add 2012-02-15

                // calculate the density matrix using read in wave functions
                // and the ncalculate the charge density on grid.
                loc.sum_bands(uhm);
                // calculate the local potential(rho) again.
                // the grid integration will do in later grid integration.


                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                // a puzzle remains here.
                // if I don't renew potential,
                // The scf_thr is very small.
                // OneElectron, Hartree and
                // Exc energy are all correct
                // except the band energy.
                //
                // solved by mohan 2010-09-10
                // there are there rho here:
                // rho1: formed by read in orbitals.
                // rho2: atomic rho, used to construct H
                // rho3: generated by after diagonalize
                // here converged because rho3 and rho1
                // are very close.
                // so be careful here, make sure
                // rho1 and rho2 are the same rho.
                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                GlobalC::pot.vr = GlobalC::pot.v_of_rho(GlobalC::CHR.rho, GlobalC::CHR.rho_core);
                GlobalC::en.delta_escf();
                if (ELEC_evolve::td_vext == 0)
                {
                    GlobalC::pot.set_vr_eff();
                }
                else
                {
                    GlobalC::pot.set_vrs_tddft(istep);
                }
            }
        }

#ifdef __MPI
        // calculate exact-exchange
        if (XC_Functional::get_func_type() == 4)
        {
            if (!GlobalC::exx_global.info.separate_loop)
            {
                GlobalC::exx_lcao.cal_exx_elec(loc, lowf.wfc_k_grid);
            }
        }
#endif

        if (INPUT.dft_plus_u)
        {
            GlobalC::dftu.cal_slater_UJ(istep, iter); // Calculate U and J if Yukawa potential is used
        }

        // (1) calculate the bands.
        // mohan add 2021-02-09
        if (GlobalV::GAMMA_ONLY_LOCAL)
        {
            ELEC_cbands_gamma::cal_bands(istep, uhm, lowf, loc.dm_gamma);
        }
        else
        {
            if (ELEC_evolve::tddft && istep >= 1 && iter > 1)
            {
                ELEC_evolve::evolve_psi(istep, uhm, lowf);
            }
            else
            {
                ELEC_cbands_k::cal_bands(istep, uhm, lowf, loc.dm_k);
            }
        }


        //		for(int ib=0; ib<GlobalV::NBANDS; ++ib)
        //		{
        //			std::cout << ib+1 << " " << GlobalC::wf.ekb[0][ib] << std::endl;
        //		}

                //-----------------------------------------------------------
                // only deal with charge density after both wavefunctions.
                // are calculated.
                //-----------------------------------------------------------
        if (GlobalV::GAMMA_ONLY_LOCAL && GlobalV::NSPIN == 2 && GlobalV::CURRENT_SPIN == 0) continue;


        if (conv_elec)
        {
            ModuleBase::timer::tick("ELEC_scf", "scf");
            return;
        }

        GlobalC::en.eband = 0.0;
        GlobalC::en.ef = 0.0;
        GlobalC::en.ef_up = 0.0;
        GlobalC::en.ef_dw = 0.0;

        // demet is included into eband.
        //if(GlobalV::DIAGO_TYPE!="selinv")
        {
            GlobalC::en.demet = 0.0;
        }

        // (2)
        GlobalC::CHR.save_rho_before_sum_band();

        // (3) sum bands to calculate charge density
        Occupy::calculate_weights();

        if (GlobalV::ocp == 1)
        {
            for (int ik = 0; ik < GlobalC::kv.nks; ik++)
            {
                for (int ib = 0; ib < GlobalV::NBANDS; ib++)
                {
                    GlobalC::wf.wg(ik, ib) = GlobalV::ocp_kb[ik * GlobalV::NBANDS + ib];
                }
            }
        }


        for (int ik = 0; ik < GlobalC::kv.nks; ++ik)
        {
            GlobalC::en.print_band(ik);
        }

        // if selinv is used, we need this to calculate the charge
        // using density matrix.
        loc.sum_bands(uhm);

#ifdef __MPI
        // add exx
        // Peize Lin add 2016-12-03
        GlobalC::en.set_exx();

        // Peize Lin add 2020.04.04
        if (XC_Functional::get_func_type() == 4)
        {
            if (GlobalC::restart.info_load.load_H && GlobalC::restart.info_load.load_H_finish && !GlobalC::restart.info_load.restart_exx)
            {
                XC_Functional::set_xc_type(GlobalC::ucell.atoms[0].xc_func);
                GlobalC::exx_lcao.cal_exx_elec(loc, lowf.wfc_k_grid);
                GlobalC::restart.info_load.restart_exx = true;
            }
        }
#endif

        // if DFT+U calculation is needed, this function will calculate
        // the local occupation number matrix and energy correction
        if (INPUT.dft_plus_u)
        {
            if (GlobalV::GAMMA_ONLY_LOCAL) GlobalC::dftu.cal_occup_m_gamma(iter, loc.dm_gamma);
            else GlobalC::dftu.cal_occup_m_k(iter, loc.dm_k);

            GlobalC::dftu.cal_energy_correction(istep);
            GlobalC::dftu.output();
        }

#ifdef __DEEPKS
        if (GlobalV::deepks_scf)
        {
            const Parallel_Orbitals* pv = lowf.ParaV;
            if (GlobalV::GAMMA_ONLY_LOCAL)
            {
                GlobalC::ld.cal_e_delta_band(loc.dm_gamma,
                    pv->trace_loc_row, pv->trace_loc_col, pv->nrow);
            }
            else
            {
                GlobalC::ld.cal_e_delta_band_k(loc.dm_k,
                    pv->trace_loc_row, pv->trace_loc_col,
                    GlobalC::kv.nks, pv->nrow, pv->ncol);
            }
        }
#endif
        // (4) mohan add 2010-06-24
        // using new charge density.
        GlobalC::en.calculate_harris(2);

        // (5) symmetrize the charge density
        Symmetry_rho srho;
        for (int is = 0; is < GlobalV::NSPIN; is++)
        {
            srho.begin(is, GlobalC::CHR, GlobalC::pw, GlobalC::Pgrid, GlobalC::symm);
        }

        // (6) compute magnetization, only for spin==2
        GlobalC::ucell.magnet.compute_magnetization();

        // resume codes!
        //-------------------------------------------------------------------------
        // this->GlobalC::LOWF.init_Cij( 0 ); // check the orthogonality of local orbital.
        // GlobalC::CHR.sum_band(); use local orbital in plane wave basis to calculate bands.
        // but must has evc first!
        //-------------------------------------------------------------------------

        // (7) calculate delta energy
        GlobalC::en.deband = GlobalC::en.delta_e();

        // (8) Mix charge density
        GlobalC::CHR.tmp_mixrho(scf_thr, 0, GlobalV::SCF_THR, iter, conv_elec);

        // Peize Lin add 2020.04.04
        if (GlobalC::restart.info_save.save_charge)
        {
            for (int is = 0; is < GlobalV::NSPIN; ++is)
            {
                GlobalC::restart.save_disk(*uhm.LM, "charge", is);
            }
        }

        // (9) Calculate new potential according to new Charge Density.

        if (conv_elec || iter == GlobalV::SCF_NMAX)
        {
            if (GlobalC::pot.out_pot < 0) //mohan add 2011-10-10
            {
                GlobalC::pot.out_pot = -2;
            }
        }

        if (!conv_elec)
        {
            // option 1
            GlobalC::pot.vr = GlobalC::pot.v_of_rho(GlobalC::CHR.rho, GlobalC::CHR.rho_core);
            GlobalC::en.delta_escf();

            // option 2
            //------------------------------
            // mohan add 2012-06-08
            // use real E_tot functional.
            //------------------------------
            /*
            GlobalC::pot.vr = GlobalC::pot.v_of_rho(GlobalC::CHR.rho_save, GlobalC::CHR.rho);
            GlobalC::en.calculate_etot();
            GlobalC::en.print_etot(conv_elec, iter, scf_thr, 0.0, GlobalV::PW_DIAG_THR, avg_iter,0);
            GlobalC::pot.vr = GlobalC::pot.v_of_rho(GlobalC::CHR.rho, GlobalC::CHR.rho_core);
            GlobalC::en.delta_escf();
            */
        }
        else
        {
            GlobalC::pot.vnew = GlobalC::pot.v_of_rho(GlobalC::CHR.rho, GlobalC::CHR.rho_core);
            //(used later for scf correction to the forces )
            GlobalC::pot.vnew -= GlobalC::pot.vr;
            GlobalC::en.descf = 0.0;
        }

        //-----------------------------------
        // output charge density for tmp
        //-----------------------------------
        for (int is = 0; is < GlobalV::NSPIN; is++)
        {
            const int precision = 3;

            std::stringstream ssc;
            ssc << GlobalV::global_out_dir << "tmp" << "_SPIN" << is + 1 << "_CHG";
            GlobalC::CHR.write_rho(GlobalC::CHR.rho_save[is], is, iter, ssc.str(), precision);//mohan add 2007-10-17

            std::stringstream ssd;

            if (GlobalV::GAMMA_ONLY_LOCAL)
            {
                ssd << GlobalV::global_out_dir << "tmp" << "_SPIN" << is + 1 << "_DM";
            }
            else
            {
                ssd << GlobalV::global_out_dir << "tmp" << "_SPIN" << is + 1 << "_DM_R";
            }
            loc.write_dm(is, iter, ssd.str(), precision);

            //LiuXh modify 20200701
            /*
            std::stringstream ssp;
            ssp << GlobalV::global_out_dir << "tmp" << "_SPIN" << is + 1 << "_POT";
            GlobalC::pot.write_potential( is, iter, ssp.str(), GlobalC::pot.vr, precision );
            */
        }

        // (10) add Vloc to Vhxc.
        if (ELEC_evolve::td_vext == 0)
        {
            GlobalC::pot.set_vr_eff();
        }
        else
        {
            GlobalC::pot.set_vrs_tddft(istep);
        }

        //time_finish=std::time(NULL);
#ifdef __MPI
        double duration = (double)(MPI_Wtime() - clock_start);
#else
        double duration = (std::chrono::system_clock::now() - clock_start).count() / CLOCKS_PER_SEC;
#endif
        //double duration_time = difftime(time_finish, time_start);
        //std::cout<<"Time_clock\t"<<"Time_time"<<std::endl;
        //std::cout<<duration<<"\t"<<duration_time<<std::endl;

        // (11) calculate the total energy.
        GlobalC::en.calculate_etot();

        // avg_iter is an useless variable in LCAO,
        // will fix this interface in future -- mohan 2021-02-10
        int avg_iter = 0;
        GlobalC::en.print_etot(conv_elec, iter, scf_thr, duration, GlobalV::PW_DIAG_THR, avg_iter);

        GlobalC::en.etot_old = GlobalC::en.etot;

        if (conv_elec || iter == GlobalV::SCF_NMAX)
        {
            //--------------------------------------
            // output charge density for converged,
            // 0 means don't need to consider iter,
            //--------------------------------------
            if (GlobalC::chi0_hilbert.epsilon)                                    // pengfei 2016-11-23
            {
                std::cout << "eta = " << GlobalC::chi0_hilbert.eta << std::endl;
                std::cout << "domega = " << GlobalC::chi0_hilbert.domega << std::endl;
                std::cout << "nomega = " << GlobalC::chi0_hilbert.nomega << std::endl;
                std::cout << "dim = " << GlobalC::chi0_hilbert.dim << std::endl;
                //std::cout <<"oband = "<<GlobalC::chi0_hilbert.oband<<std::endl;
                GlobalC::chi0_hilbert.wfc_k_grid = lowf.wfc_k_grid;
                GlobalC::chi0_hilbert.Chi();
            }

            for (int is = 0; is < GlobalV::NSPIN; is++)
            {
                const int precision = 3;

                std::stringstream ssc;
                ssc << GlobalV::global_out_dir << "SPIN" << is + 1 << "_CHG";
                GlobalC::CHR.write_rho(GlobalC::CHR.rho_save[is], is, 0, ssc.str());//mohan add 2007-10-17

                std::stringstream ssd;
                if (GlobalV::GAMMA_ONLY_LOCAL)
                {
                    ssd << GlobalV::global_out_dir << "SPIN" << is + 1 << "_DM";
                }
                else
                {
                    ssd << GlobalV::global_out_dir << "SPIN" << is + 1 << "_DM_R";
                }
                loc.write_dm(is, 0, ssd.str(), precision);

                if (GlobalC::pot.out_pot == 1) //LiuXh add 20200701
                {
                    std::stringstream ssp;
                    ssp << GlobalV::global_out_dir << "SPIN" << is + 1 << "_POT";
                    GlobalC::pot.write_potential(is, 0, ssp.str(), GlobalC::pot.vr_eff, precision);
                }

                //LiuXh modify 20200701
                /*
                //fuxiang add 2017-03-15
                std::stringstream sse;
                sse << GlobalV::global_out_dir << "SPIN" << is + 1 << "_DIPOLE_ELEC";
                GlobalC::CHR.write_rho_dipole(GlobalC::CHR.rho_save, is, 0, sse.str());
                */
            }

            if (conv_elec)
            {
                GlobalV::ofs_running << "\n charge density convergence is achieved" << std::endl;
                GlobalV::ofs_running << " final etot is " << GlobalC::en.etot * ModuleBase::Ry_to_eV << " eV" << std::endl;
            }

            if (GlobalV::OUT_LEVEL != "m")
            {
                print_eigenvalue(GlobalV::ofs_running);
            }

            if (conv_elec)
            {
                //xiaohui add "OUT_LEVEL", 2015-09-16
                if (GlobalV::OUT_LEVEL != "m") GlobalV::ofs_running << std::setprecision(16);
                if (GlobalV::OUT_LEVEL != "m") GlobalV::ofs_running << " EFERMI = " << GlobalC::en.ef * ModuleBase::Ry_to_eV << " eV" << std::endl;
                if (GlobalV::OUT_LEVEL == "ie")
                {
                    GlobalV::ofs_running << " " << GlobalV::global_out_dir << " final etot is " << GlobalC::en.etot * ModuleBase::Ry_to_eV << " eV" << std::endl;
                }
            }
            else
            {
                GlobalV::ofs_running << " !! convergence has not been achieved @_@" << std::endl;
                if (GlobalV::OUT_LEVEL == "ie" || GlobalV::OUT_LEVEL == "m") //xiaohui add "m" option, 2015-09-16
                    std::cout << " !! CONVERGENCE HAS NOT BEEN ACHIEVED !!" << std::endl;
            }

            if (conv_elec || iter == GlobalV::SCF_NMAX)
            {
#ifdef __DEEPKS
                //calculating deepks correction to bandgap
                //and save the results

                if (GlobalV::deepks_out_labels)	//caoyu add 2021-06-04
                {
                    int nocc = GlobalC::CHR.nelec / 2;
                    if (GlobalV::deepks_bandgap)
                    {
                        if (GlobalV::GAMMA_ONLY_LOCAL)
                        {
                            GlobalC::ld.save_npy_o(GlobalC::wf.ekb[0][nocc] - GlobalC::wf.ekb[0][nocc - 1], "o_tot.npy");
                        }
                        else
                        {
                            double homo = GlobalC::wf.ekb[0][nocc - 1];
                            double lumo = GlobalC::wf.ekb[0][nocc];
                            for (int ik = 1; ik < GlobalC::kv.nks; ik++)
                            {
                                if (homo < GlobalC::wf.ekb[ik][nocc - 1])
                                {
                                    homo = GlobalC::wf.ekb[ik][nocc - 1];
                                    GlobalC::ld.h_ind = ik;
                                }
                                if (lumo > GlobalC::wf.ekb[ik][nocc])
                                {
                                    lumo = GlobalC::wf.ekb[ik][nocc];
                                    GlobalC::ld.l_ind = ik;
                                }
                            }
                            GlobalC::ld.save_npy_o(lumo - homo - GlobalC::ld.o_delta, "o_tot.npy");
                            GlobalV::ofs_running << " HOMO index is " << GlobalC::ld.h_ind << std::endl;
                            GlobalV::ofs_running << " HOMO energy " << homo << std::endl;
                            GlobalV::ofs_running << " LUMO index is " << GlobalC::ld.l_ind << std::endl;
                            GlobalV::ofs_running << " LUMO energy " << lumo << std::endl;
                        }
                    }

                    GlobalC::ld.save_npy_e(GlobalC::en.etot, "e_tot.npy");
                    if (GlobalV::deepks_scf)
                    {
                        GlobalC::ld.save_npy_e(GlobalC::en.etot - GlobalC::ld.E_delta, "e_base.npy");//ebase :no deepks E_delta including
                        if (GlobalV::deepks_bandgap)
                        {
                            int nocc = GlobalC::CHR.nelec / 2;

                            ModuleBase::matrix wg_hl;
                            if (GlobalV::GAMMA_ONLY_LOCAL)
                            {
                                wg_hl.create(GlobalV::NSPIN, GlobalV::NBANDS);

                                for (int is = 0; is < GlobalV::NSPIN; is++)
                                {
                                    for (int ib = 0; ib < GlobalV::NBANDS; ib++)
                                    {
                                        wg_hl(is, ib) = 0.0;
                                        if (ib == nocc - 1)
                                            wg_hl(is, ib) = -1.0;
                                        else if (ib == nocc)
                                            wg_hl(is, ib) = 1.0;
                                    }
                                }

                                std::vector<ModuleBase::matrix> dm_bandgap_gamma;
                                dm_bandgap_gamma.resize(GlobalV::NSPIN);
                                loc.cal_dm(wg_hl, lowf.wfc_gamma, dm_bandgap_gamma);


                                GlobalC::ld.cal_orbital_precalc(dm_bandgap_gamma,
                                    GlobalC::ucell.nat,
                                    GlobalC::ucell,
                                    GlobalC::ORB,
                                    GlobalC::GridD,
                                    *lowf.ParaV);

                                GlobalC::ld.save_npy_orbital_precalc(GlobalC::ucell.nat);

                                GlobalC::ld.cal_o_delta(dm_bandgap_gamma, *lowf.ParaV);
                                GlobalC::ld.save_npy_o(GlobalC::wf.ekb[0][nocc] - GlobalC::wf.ekb[0][nocc - 1] - GlobalC::ld.o_delta, "o_base.npy");

                            }
                            else //multi-k bandgap label
                            {
                                wg_hl.create(GlobalC::kv.nks, GlobalV::NBANDS);

                                for (int ik = 0; ik < GlobalC::kv.nks; ik++) {
                                    for (int ib = 0; ib < GlobalV::NBANDS; ib++) {
                                        wg_hl(ik, ib) = 0.0;

                                        if (ik == GlobalC::ld.h_ind && ib == nocc - 1)
                                            wg_hl(ik, ib) = -1.0;
                                        else if (ik == GlobalC::ld.l_ind && ib == nocc)
                                            wg_hl(ik, ib) = 1.0;
                                    }
                                }
                                std::vector<ModuleBase::ComplexMatrix> dm_bandgap_k;
                                dm_bandgap_k.resize(GlobalC::kv.nks);
                                loc.cal_dm(wg_hl, lowf.wfc_k, dm_bandgap_k);
                                GlobalC::ld.cal_o_delta_k(dm_bandgap_k, *lowf.ParaV, GlobalC::kv.nks);

                                GlobalC::ld.cal_orbital_precalc_k(dm_bandgap_k,
                                    GlobalC::ucell.nat,
                                    GlobalC::kv.nks,
                                    GlobalC::kv.kvec_d,
                                    GlobalC::ucell,
                                    GlobalC::ORB,
                                    GlobalC::GridD,
                                    *lowf.ParaV);
                                GlobalC::ld.save_npy_orbital_precalc(GlobalC::ucell.nat);

                                GlobalC::ld.cal_o_delta_k(dm_bandgap_k, *lowf.ParaV, GlobalC::kv.nks);
                                GlobalC::ld.save_npy_o(GlobalC::wf.ekb[GlobalC::ld.l_ind][nocc] - GlobalC::wf.ekb[GlobalC::ld.h_ind][nocc - 1] - GlobalC::ld.o_delta, "o_base.npy");
                            }
                        }
                    }
                    else //deepks_scf = 0; base calculation
                    {
                        GlobalC::ld.save_npy_e(GlobalC::en.etot, "e_base.npy");  // no scf, e_tot=e_base
                        if (GlobalV::deepks_bandgap)
                        {
                            if (GlobalV::GAMMA_ONLY_LOCAL)
                            {
                                GlobalC::ld.save_npy_o(GlobalC::wf.ekb[0][nocc] - GlobalC::wf.ekb[0][nocc - 1], "o_base.npy");  // no scf, o_tot=o_base
                            }
                            else
                            {
                                GlobalC::ld.save_npy_o(GlobalC::wf.ekb[GlobalC::ld.l_ind][nocc] - GlobalC::wf.ekb[GlobalC::ld.h_ind][nocc - 1], "o_base.npy");
                            }
                        }
                    }
                }
#endif
            }
            //			ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running,"ELECTRONS CONVERGED!");
            ModuleBase::timer::tick("ELEC_scf", "scf");
            return;
        }
    }

    ModuleBase::timer::tick("ELEC_scf", "scf");
    return;
}


void ELEC_scf::init_mixstep_final_scf(void)
{
    ModuleBase::TITLE("ELEC_scf", "init_mixstep_final_scf");

    GlobalC::CHR.irstep = 0;
    GlobalC::CHR.idstep = 0;
    GlobalC::CHR.totstep = 0;

    return;
}

