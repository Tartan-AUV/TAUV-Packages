from typing import Optional, Type
from missions.mission import Mission
import missions

def get_mission_by_name(name: str) -> Optional[Type[Mission]]:
    missions_by_name = {
        "kf_transdec_23": missions.kf_transdec_23_buoy_search.KFTransdec23,
        "kf_transdec_23_survey": missions.kf_transdec_23_survey.KFTransdec23,
        "kf_prequal_24": missions.kf_prequal_24.KFPrequal24,
         "irvine_semis": missions.kf_irvine_semis1.KFIrvineSemis1,
        "kf_buoy_dive_24": missions.kf_buoy_dive_24.KFBuoyDive24,
        "irvine_semis_2": missions.kf_irvine_semis2.KFIrvineSemis2,
        "irvine_semis_3": missions.kf_irvine_semis2_2.KFIrvineSemis3,
        "irvine_finals": missions.kf_irvine_finals_dead_reckon.KFIrvineFinals
    }
    return missions_by_name.get(name)
