def check_claim(L_um, target_nm=1550, tol_nm=2):
    # resonances: lambda_m = 2*L*1000/m   (n = 1)
    m = round(2 * L_um * 1000 / target_nm)
    # find the m whose lambda_m is closest to target_nm
    lambda_m = 2 * L_um * 1000 / m if m != 0 else float('inf')
    distance_nm = abs(lambda_m - target_nm)
    passed = distance_nm <= tol_nm
    # return (nearest_resonance_nm, distance_nm, passed: bool)
    return lambda_m, distance_nm, passed