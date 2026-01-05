from QMRegionSelector import QMRegionSelector

selector = QMRegionSelector("qm_region.json")
selector.getRefQM()
selector.write_ref_outputs()
selector.getGroundCharge()
selector.getExcitedCharge()
selector.getChargeShiftPerResidue()
selector.getCSARegion()
