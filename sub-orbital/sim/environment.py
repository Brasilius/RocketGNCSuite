"""ISA atmosphere model (0–86 km) and altitude-dependent gravity."""
import numpy as np


def atmosphere(altitude_m: float) -> tuple[float, float, float]:
    """Return (density kg/m³, pressure Pa, temperature K) at given altitude."""
    alt = max(0.0, float(altitude_m))
    g0 = 9.80665
    R_air = 287.058

    if alt <= 11_000:
        T = 288.15 - 0.0065 * alt
        P = 101_325.0 * (T / 288.15) ** (g0 / (R_air * 0.0065))
    elif alt <= 20_000:
        T = 216.65
        P = 22_632.1 * np.exp(-g0 * (alt - 11_000) / (R_air * T))
    elif alt <= 32_000:
        T = 216.65 + 0.001 * (alt - 20_000)
        P = 5_474.89 * (T / 216.65) ** (-g0 / (R_air * 0.001))
    elif alt <= 47_000:
        T = 228.65 + 0.0028 * (alt - 32_000)
        P = 868.019 * (T / 228.65) ** (-g0 / (R_air * 0.0028))
    elif alt <= 86_000:
        T = 270.65
        P = 110.906 * np.exp(-g0 * (alt - 47_000) / (R_air * T))
    else:
        T = 186.87
        P = 3.7338 * np.exp(-g0 * (alt - 86_000) / (R_air * T))

    return P / (R_air * T), P, T


def gravity(altitude_m: float) -> float:
    """Gravity magnitude (m/s²) accounting for altitude."""
    R_earth = 6_371_000.0
    return 9.80665 * (R_earth / (R_earth + max(0.0, altitude_m))) ** 2
