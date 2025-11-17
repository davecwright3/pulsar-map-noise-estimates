"""Add noise to pulsar timing model using Pint.

Taken from pint_pal and modified to accept a dictionary as input.
"""

import numpy as np
import pint.models as pm
from loguru import logger as log
from pint.models.parameter import maskParameter


def convert_to_RNAMP(value):
    """Utility function to convert enterprise RN amplitude to tempo2/PINT parfile RN amplitude."""
    return (86400.0 * 365.24 * 1e6) / (2.0 * np.pi * np.sqrt(3.0)) * 10**value


def add_noise_to_model(
    model,
    noise_dict,
):
    """Add WN, RN, DMGP, ChromGP, and SW parameters to timing model.

    Parameters
    ----------
    model: PINT (or tempo2) timing model
    noise_dict: dict[str, array_like]
        Dictionary with noise parameters to use in timing model update

    Returns
    -------
    model: New timing model which includes WN and RN (and potentially dmgp, chrom_gp, and solar wind) parameters

    """
    # Assume results are in current working directory if not specified
    # Create the maskParameter for EFACS
    efac_params = []
    equad_params = []
    ecorr_params = []

    efac_idx = 1
    equad_idx = 1
    ecorr_idx = 1

    psr_name = next(iter(noise_dict.keys())).split("_")[0]
    noise_pars = np.array(list(noise_dict.keys()))
    wn_dict = {
        key: val
        for key, val in noise_dict.items()
        if "efac" in key or "equad" in key or "ecorr" in key
    }
    for key, val in wn_dict.items():
        if "_efac" in key:
            param_name = key.split("_efac")[0].split(psr_name)[1][1:]

            tp = maskParameter(
                name="EFAC",
                index=efac_idx,
                key="-f",
                key_value=param_name,
                value=val,
                units="",
                convert_tcb2tdb=False,
            )
            efac_params.append(tp)
            efac_idx += 1

        # See https://github.com/nanograv/enterprise/releases/tag/v3.3.0
        # ..._t2equad uses PINT/Tempo2/Tempo convention, resulting in total variance EFAC^2 x (toaerr^2 + EQUAD^2)
        elif "_t2equad" in key:
            param_name = (
                key.split("_t2equad")[0].split(psr_name)[1].split("_log10")[0][1:]
            )

            tp = maskParameter(
                name="EQUAD",
                index=equad_idx,
                key="-f",
                key_value=param_name,
                value=10**val / 1e-6,
                units="us",
                convert_tcb2tdb=False,
            )
            equad_params.append(tp)
            equad_idx += 1

        # ..._equad uses temponest convention; generated with enterprise pre-v3.3.0
        elif "_equad" in key:
            param_name = (
                key.split("_equad")[0].split(psr_name)[1].split("_log10")[0][1:]
            )

            tp = maskParameter(
                name="EQUAD",
                index=equad_idx,
                key="-f",
                key_value=param_name,
                value=10**val / 1e-6,
                units="us",
                convert_tcb2tdb=False,
            )
            equad_params.append(tp)
            equad_idx += 1

        elif "_ecorr" in key:
            param_name = (
                key.split("_ecorr")[0].split(psr_name)[1].split("_log10")[0][1:]
            )

            tp = maskParameter(
                name="ECORR",
                index=ecorr_idx,
                key="-f",
                key_value=param_name,
                value=10**val / 1e-6,
                units="us",
                convert_tcb2tdb=False,
            )
            ecorr_params.append(tp)
            ecorr_idx += 1

    # Create white noise components and add them to the model
    ef_eq_comp = pm.ScaleToaError()
    ef_eq_comp.remove_param(param="EFAC1")
    ef_eq_comp.remove_param(param="EQUAD1")
    ef_eq_comp.remove_param(param="TNEQ1")
    for efac_param in efac_params:
        ef_eq_comp.add_param(param=efac_param, setup=True)
    for equad_param in equad_params:
        ef_eq_comp.add_param(param=equad_param, setup=True)
    model.add_component(ef_eq_comp, validate=True, force=True)

    if len(ecorr_params) > 0:
        ec_comp = pm.EcorrNoise()
        ec_comp.remove_param("ECORR1")
        for ecorr_param in ecorr_params:
            ec_comp.add_param(param=ecorr_param, setup=True)
        model.add_component(ec_comp, validate=True, force=True)

    log.info(f"Including red noise for {psr_name}")
    # Add the ML RN parameters to their component
    rn_comp = pm.PLRedNoise()

    rn_keys = np.array([key for key, val in noise_dict.items() if "_red_" in key])
    rn_comp.RNAMP.quantity = convert_to_RNAMP(
        noise_dict[psr_name + "_red_noise_log10_A"],
    )
    rn_comp.RNIDX.quantity = -1 * noise_dict[psr_name + "_red_noise_gamma"]
    # Add red noise to the timing model
    model.add_component(rn_comp, validate=True, force=True)

    # Check to see if dm noise is present
    dm_pars = [key for key in noise_pars if "_dm_gp" in key]
    if len(dm_pars) > 0:
        ###### POWERLAW DM NOISE ######
        if f"{psr_name}_dm_gp_log10_A" in dm_pars:
            # dm_bf = model_utils.bayes_fac(noise_core(rn_amp_nm), ntol=1, logAmax=-11, logAmin=-20)[0]
            # log.info(f"The SD Bayes factor for dm noise in this pulsar is: {dm_bf}")
            log.info("Adding Powerlaw DM GP noise as PLDMNoise to par file")
            # Add the ML RN parameters to their component
            dm_comp = pm.noise_model.PLDMNoise()
            dm_keys = np.array(
                [key for key, val in noise_dict.items() if "_red_" in key],
            )
            dm_comp.TNDMAMP.quantity = convert_to_RNAMP(
                noise_dict[psr_name + "_dm_gp_log10_A"],
            )
            dm_comp.TNDMGAM.quantity = -1 * noise_dict[psr_name + "_dm_gp_gamma"]
            ##### FIXMEEEEEEE : need to figure out some way to softcode this
            dm_comp.TNDMC.quantitity = 100
            # Add red noise to the timing model
            model.add_component(dm_comp, validate=True, force=True)
        ###### FREE SPECTRAL (WaveX) DM NOISE ######
        elif f"{psr_name}_dm_gp_log10_rho_0" in dm_pars:
            log.info("Adding Free Spectral DM GP as DMWaveXnoise to par file")
            raise NotImplementedError("DMWaveXNoise not yet implemented")

    # Check to see if higher order chromatic noise is present
    chrom_pars = [key for key in noise_pars if "_chrom_gp" in key]
    if len(chrom_pars) > 0:
        ###### POWERLAW CHROMATIC NOISE ######
        if f"{psr_name}_chrom_gp_log10_A" in chrom_pars:
            log.info("Adding Powerlaw CHROM GP noise as PLCMNoise to par file")
            # Add the ML RN parameters to their component
            chrom_comp = pm.noise_model.PLCMNoise()
            # chrom_keys = np.array([key for key, val in noise_dict.items() if "_chrom_gp_" in key])
            chrom_comp.TNCMAMP.quantity = convert_to_RNAMP(
                noise_dict[psr_name + "_chrom_gp_log10_A"],
            )
            chrom_comp.TNCMGAM.quantity = -1 * noise_dict[psr_name + "_chrom_gp_gamma"]
            ##### FIXMEEEEEEE : need to figure out some way to softcode this
            chrom_comp.TNCMC.quantitity = 100
            # Add red noise to the timing model
            model.add_component(chrom_comp, validate=True, force=True)
        ###### FREE SPECTRAL (WaveX) DM NOISE ######
        elif f"{psr_name}_chrom_gp_log10_rho_0" in chrom_pars:
            log.info("Adding Free Spectral CHROM GP as CMWaveXnoise to par file")
            raise NotImplementedError("CMWaveXNoise not yet implemented")

    # Check to see if solar wind is present
    sw_pars = [key for key in noise_pars if "sw_r2" in key]
    if len(sw_pars) > 0:
        log.info("Adding Solar Wind Dispersion to par file")
        all_components = Component.component_types
        noise_class = all_components["SolarWindDispersion"]
        noise = noise_class()  # Make the dispersion instance.
        model.add_component(noise, validate=False, force=False)
        # add parameters
        if f"{psr_name}_n_earth" in sw_pars:
            model["NE_SW"].quantity = noise_dict[f"{psr_name}_n_earth"]
            model["NE_SW"].frozen = True
        if f"{psr_name}_sw_gp_log10_A" in sw_pars:
            sw_comp = pm.noise_model.PLSWNoise()
            sw_comp.TNSWAMP.quantity = convert_to_RNAMP(
                noise_dict[f"{psr_name}_sw_gp_log10_A"],
            )
            sw_comp.TNSWAMP.frozen = True
            sw_comp.TNSWGAM.quantity = -1.0 * noise_dict[f"{psr_name}_sw_gp_gamma"]
            sw_comp.TNSWGAM.frozen = True
            # FIXMEEEEEEE : need to figure out some way to softcode this
            sw_comp.TNSWC.quantity = 10
            sw_comp.TNSWC.frozen = True
            model.add_component(sw_comp, validate=False, force=True)
        if f"{psr_name}_sw_gp_log10_rho" in sw_pars:
            raise NotImplementedError(
                "Solar Wind Dispersion free spec GP not yet implemented",
            )

    # Setup and validate the timing model to ensure things are correct
    model.setup()
    model.validate()

    return model
