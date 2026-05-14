"""1D Surface Brightness Profile rendering for GALFIT comparison plots.

Provides isophote-based radial profile extraction and matplotlib rendering
of data vs model surface brightness with a residual sub-panel.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from photutils.isophote import EllipseSample, Ellipse
    from photutils.isophote.geometry import EllipseGeometry
    HAS_PHOTUTILS = True
except ImportError:
    HAS_PHOTUTILS = False

DEFAULT_COLORS = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


def parse_photometry_params(param_file: str) -> tuple[float, float]:
    """Parse zeropoint (J) and plate scale (K) from a GALFIT parameter file."""
    zeropoint, pltscale = 21.097, 0.750
    try:
        with open(param_file) as f:
            for line in f:
                s = line.strip()
                if s.startswith("J)"):
                    zeropoint = float(s.split()[1])
                elif s.startswith("K)"):
                    pltscale = float(s.split()[1])
    except Exception:
        pass
    return zeropoint, pltscale


def fit_data_isophotes(image_data, x_center, y_center,
                       pa_deg=None, eps=None, sma_max=None, mask=None):
    """Fit isophotes using photutils. Returns IsophoteList or None."""
    if not HAS_PHOTUTILS:
        return None
    if mask is not None:
        image_data = np.ma.array(image_data, mask=mask > 0)
    # Replace NaN with 0 to prevent photutils sampling failures at image edges
    if np.any(np.isnan(image_data)):
        image_data = np.nan_to_num(image_data, nan=0.0)
    maxsma = sma_max or min(1200.0, max(image_data.shape) * 0.45)
    ny, nx = image_data.shape
    edge_dist = min(x_center, y_center, nx - x_center, ny - y_center)
    maxsma = min(maxsma, edge_dist * 0.9)

    strategies = [
        {"sma0": 10.0, "integrmode": "median", "step": 0.2},
        {"sma0": 5.0, "integrmode": "bilinear", "step": 0.15},
        {"sma0": 20.0, "integrmode": "median", "step": 0.2},
        {"sma0": 3.0, "integrmode": "bilinear", "step": 0.1},
    ]
    pa_rad = np.radians(pa_deg) if pa_deg is not None else None
    eps_val = eps if eps is not None else None

    best_result = None
    for strat in strategies:
        for pa in [pa_rad, 0.0, np.radians(90)]:
            if pa is None:
                pa = 0.0
            for ev in [eps_val, 0.1, 0.3, 0.5] if eps_val is None else [eps_val]:
                try:
                    geo = EllipseGeometry(x_center, y_center, strat["sma0"],
                                          ev, pa)
                    ellipse = Ellipse(image_data, geometry=geo)
                    isolist = ellipse.fit_image(
                        sma0=strat["sma0"], minsma=1.0, maxsma=maxsma,
                        step=strat["step"], linear=False,
                        integrmode=strat["integrmode"], sclip=3.0, nclip=3,
                    )
                    if len(isolist) > 5:
                        return isolist
                    if best_result is None or len(isolist) > len(best_result):
                        best_result = isolist
                except Exception:
                    continue
    return best_result


def extract_profile(image_data, geometry, x_offset=0, y_offset=0, mask=None):
    """Extract 1D radial profile using pre-fitted isophote geometry.

    Args:
        image_data: 2D image array.
        geometry: List of (sma, eps, pa_deg, x0, y0) tuples.
        x_offset: Offset to subtract from isophote x-center.
        y_offset: Offset to subtract from isophote y-center.
        mask: 2D mask array (mask>0 = bad pixel).

    Returns:
        (sma_array, intensity_array) as numpy arrays.
    """
    if not HAS_PHOTUTILS:
        return np.array([]), np.array([])
    if mask is not None:
        image_data = np.ma.array(image_data, mask=mask > 0)
    sma_arr, intensity_arr = [], []
    for sma, eps, pa_deg, x0, y0 in geometry:
        if sma < 1:
            continue
        try:
            sample = EllipseSample(image_data, sma,
                                   x0=x0 - x_offset, y0=y0 - y_offset,
                                   eps=eps, position_angle=np.radians(pa_deg))
            s = sample.extract()
            if len(s) == 0:
                continue
            intensities = s[2]
            if len(intensities) > 0:
                med = np.median(intensities)
                if med > 1e-5:
                    sma_arr.append(sma)
                    intensity_arr.append(med)
        except Exception:
            continue
    return np.array(sma_arr), np.array(intensity_arr)


def intensity_to_sb(intensity, zeropoint, pixscale):
    """Convert intensity to surface brightness (mag/arcsec²)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return -2.5 * np.log10(intensity / pixscale ** 2) + zeropoint


def render_sb_profile(ax_main, ax_resid, original_data, model_data,
                      param_file, components, fit_region,
                      comp_images=None, comp_types=None, mask=None):
    """Render 1D SB profile onto a pair of (main, residual) axes.

    Fits isophotes on the original data, extracts profiles for both data and
    model, converts to mag/arcsec², then draws scatter/line plots plus a
    log-scale inset and a Δμ residual panel.

    If photutils is unavailable or isophote fitting fails, a placeholder
    message is drawn instead.

    Args:
        ax_main: Matplotlib Axes for the main SB profile.
        ax_resid: Matplotlib Axes for the residual (Δμ) panel (sharex with ax_main).
        original_data: 2D original image array (cropped to fit region).
        model_data: 2D model image array (same shape as original_data).
        param_file: Path to GALFIT parameter file for zeropoint/plate scale.
        components: List of component dicts from parse_components (may be None).
        fit_region: (xmin, xmax, ymin, ymax) in 1-indexed pixels, or None.
    """
    if not HAS_PHOTUTILS:
        ax_main.text(0.5, 0.5, 'SB Profile unavailable (photutils not installed)',
                     ha='center', va='center', transform=ax_main.transAxes,
                     fontsize=11, color='gray')
        _style_resid_axes(ax_resid)
        return

    if param_file is None or model_data is None:
        ax_main.text(0.5, 0.5, 'SB Profile unavailable (missing data)',
                     ha='center', va='center', transform=ax_main.transAxes,
                     fontsize=11, color='gray')
        _style_resid_axes(ax_resid)
        return

    zeropoint, pltscale = parse_photometry_params(param_file)

    # Compute center in cropped-image 0-indexed coordinates
    x_cen = original_data.shape[1] / 2.0
    y_cen = original_data.shape[0] / 2.0
    init_pa, init_eps = None, None
    if components:
        c0 = components[0]
        if fit_region is not None:
            x_cen = c0["x"] - fit_region[0]
            y_cen = c0["y"] - fit_region[2]
        if c0.get("pa"):
            init_pa = c0["pa"]
        if 0 < c0.get("ba", 1) < 1:
            init_eps = 1.0 - c0["ba"]

    sma_max = min(original_data.shape) * 0.45
    isolist = fit_data_isophotes(original_data, x_cen, y_cen,
                                  pa_deg=init_pa, eps=init_eps,
                                  sma_max=sma_max, mask=mask)
    if isolist is None or len(isolist) == 0:
        ax_main.text(0.5, 0.5, 'SB Profile unavailable (isophote fitting failed)',
                     ha='center', va='center', transform=ax_main.transAxes,
                     fontsize=11, color='gray')
        _style_resid_axes(ax_resid)
        return

    # Data profile directly from isophote list
    sma_data = isolist.sma
    intens_data = isolist.intens
    mu_data = intensity_to_sb(intens_data, zeropoint, pltscale)
    valid = np.isfinite(mu_data) & (intens_data > 0)
    sma_data = sma_data[valid]
    mu_data = mu_data[valid]

    # Model profile using same geometry
    geometry = [(iso.sma, iso.eps, np.degrees(iso.pa), iso.x0, iso.y0)
                for iso in isolist if iso.valid]
    sma_model, intens_model = extract_profile(model_data, geometry, mask=mask)
    mu_model = intensity_to_sb(intens_model, zeropoint, pltscale)

    # Main SB panel
    ax_main.scatter(sma_data, mu_data, s=8, facecolors='none',
                    edgecolors='black', linewidths=0.4, zorder=5, label='Data')
    ax_main.plot(sma_model, mu_model, 'r--', linewidth=1.2,
                 zorder=4, label='Total Model')

    # Inset with log-scaled x-axis
    ax_inset = ax_main.inset_axes([0.55, 0.5, 0.4, 0.45])
    ax_inset.scatter(sma_data, mu_data, s=6, facecolors='none',
                     edgecolors='black', linewidths=0.3, zorder=5)
    ax_inset.plot(sma_model, mu_model, 'r--', linewidth=1.0, zorder=4)

    # Component profiles (image-based from GALFIT subcomps)
    if comp_images and comp_types:
        comp_fluxes = [np.nansum(img) for img in comp_images]
        total_model_flux = np.sum(comp_fluxes)
        comp_fractions = [f / total_model_flux if total_model_flux > 0 else 0
                          for f in comp_fluxes]

        for i, (comp_img, comp_type) in enumerate(zip(comp_images, comp_types)):
            sma_c, intens_c = extract_profile(comp_img, geometry, mask=mask)
            if len(sma_c) == 0:
                continue
            mu_c = intensity_to_sb(intens_c, zeropoint, pltscale)

            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            if comp_type.lower() == 'sersic' and components and i < len(components):
                n_val = components[i].get('n')
                label = f'sersic(n={n_val:.2f}) {comp_fractions[i]:.3f}' if n_val is not None else f'sersic {comp_fractions[i]:.3f}'
            else:
                label = f'{comp_type} {comp_fractions[i]:.3f}'
            ax_main.plot(sma_c, mu_c, '-', color=color, linewidth=1.2,
                         zorder=3, label=label)
            ax_inset.plot(sma_c, mu_c, '-', color=color, linewidth=1.0)

    ax_inset.set_xscale('log')
    ax_inset.set_xlim(sma_data[sma_data > 0].min() * 0.8, sma_data.max() * 1.1)
    ax_inset.invert_yaxis()
    ax_inset.tick_params(axis='both', which='major', labelsize=9)
    ax_inset.grid(True, which='both', alpha=0.1, linestyle='--')

    ax_main.set_ylabel(r'Surface Brightness [mag arcsec$^{-2}$]', fontsize=11)
    ax_main.invert_yaxis()
    ax_main.set_xlim(0, sma_data.max() * 1.05)
    # Move legend to the lower right corner inside the plot
    ax_main.legend(loc='lower right', fontsize=9,
                   frameon=True, fancybox=True, framealpha=0.7)
    ax_main.set_title('1D Surface Brightness Profile', fontsize=11)
    ax_main.grid(True, which='both', alpha=0.1, linestyle='--')
    ax_main.tick_params(labelbottom=False)

    # Residual panel: Δμ = μ_data − μ_model
    if len(sma_model) > 2:
        from scipy.interpolate import interp1d
        common_min = max(sma_data.min(), sma_model.min())
        common_max = min(sma_data.max(), sma_model.max())
        cmask = (sma_data >= common_min) & (sma_data <= common_max)
        sma_common = sma_data[cmask]
        mu_data_c = mu_data[cmask]
        model_interp = interp1d(sma_model, mu_model,
                                kind='linear', bounds_error=False,
                                fill_value=np.nan)
        residual = mu_data_c - model_interp(sma_common)
        vresid = np.isfinite(residual)
        if np.any(vresid):
            ax_resid.axhline(0, color='gray', linewidth=0.8)
            ax_resid.scatter(sma_common[vresid], residual[vresid],
                             s=8, facecolors='none',
                             edgecolors='black', linewidths=0.5)
            ax_resid.set_ylim(-0.5, 0.5)

    _style_resid_axes(ax_resid)


def _style_resid_axes(ax):
    """Apply shared styling to the residual axes."""
    ax.set_ylabel(r'$\Delta\mu$ (Data $-$ Model)', fontsize=11)
    ax.set_xlabel(r'Semi-major Axis [pixels]', fontsize=11)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
