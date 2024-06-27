#include "center2_orb.h"
#include "module_base/tool_quit.h"

int Center2_Orb::get_rmesh(const double& R1, const double& R2, const double dr)
{
    int rmesh = static_cast<int>((R1 + R2) / dr) + 5;
    // mohan update 2009-09-08 +1 ==> +5
    // considering interpolation or so on...
    if (rmesh % 2 == 0)
        rmesh++;

    if (rmesh <= 0)
    {
        // GlobalV::ofs_warning << "\n R1 = " << R1 << " R2 = " << R2;
        // GlobalV::ofs_warning << "\n rmesh = " << rmesh;
        std::cout << "\n R1 = " << R1 << " R2 = " << R2;
        std::cout << "\n rmesh = " << rmesh;
        ModuleBase::WARNING_QUIT("Center2_Orb::get_rmesh", "rmesh <= 0");
    }
    return rmesh;
}

