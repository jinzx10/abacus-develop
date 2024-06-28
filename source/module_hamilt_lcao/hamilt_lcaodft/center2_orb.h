//==========================================================
// AUTHOR : Peize Lin
// DATE : 2016-01-24
//==========================================================

#ifndef CENTER2_ORB_H
#define CENTER2_ORB_H

class Center2_Orb
{
  public:
    class Orb11;
    class Orb21;
    class Orb22;

    // The following functions used to be in module_ao/ORB_table_phi.h

    static int get_rmesh(const double& R1, const double& R2, const double dr);
};

#endif // CENTER2_ORB_H
