import numpy as np

num_dummies_p = 10
num_dummies_z = 5
num_particles = 40
K_nearest = 6

kg, kf, h_physics = 2.5, 3, 0.1
h_grad = 1e-4
nudge_factor = 0.05
G_dummy_attraction = 1.2  
G_dummy_smell = 1.0        
kf_dummy = 5.0             

np.random.seed(42)

p_p = np.random.uniform(-12, 12, (num_particles, 1)) + 1j * np.random.uniform(-12, 12, (num_particles, 1))
v_p = np.zeros(p_p.shape, dtype=np.complex128)
d_p = np.random.uniform(-10, 10, (num_dummies_p, 1)) + 1j * np.random.uniform(-10, 10, (num_dummies_p, 1))
v_dp = np.zeros(d_p.shape, dtype=np.complex128)
poles = []

p_z = np.random.uniform(-12, 12, (num_particles, 1)) + 1j * np.random.uniform(-12, 12, (num_particles, 1))
v_z = np.zeros(p_z.shape, dtype=np.complex128)
d_z = np.random.uniform(-10, 10, (num_dummies_z, 1)) + 1j * np.random.uniform(-10, 10, (num_dummies_z, 1))
v_dz = np.zeros(d_z.shape, dtype=np.complex128)
zeros = []

def N(s):
    return s - 4

def D(s): 
    return (s**2 + 6*s + 25) * (s**2 - 12*s + 40) * (s + 10)

def cost(s, found_list, func):
    val = np.log10(np.abs(func(s)) + 1e-15)
    for f in found_list:
        val -= 2.0 * np.log10(np.abs(s - f) + 1e-15)
    return val

def gradCost(p_in, found_list, func):
    h = h_grad
    p_arr = np.array(p_in).reshape(-1, 1)
    delx = (cost(p_arr + h, found_list, func) - cost(p_arr - h, found_list, func)) / (2*h)
    dely = (cost(p_arr + 1j*h, found_list, func) - cost(p_arr - 1j*h, found_list, func)) / (2*h)
    g = delx + 1j * dely
    mag = np.abs(g)
    return np.where(mag > 1e-9, g / mag, 0)

i = 0
while i < 30000:
    g_pp = gradCost(p_p, poles, D)
    v_p = (v_p + (-g_pp * kg) * h_physics) * np.exp(-h_physics * kf)
    p_p = p_p + (v_p * h_physics) + (g_pp * nudge_factor)
    p_p.real, p_p.imag = np.clip(p_p.real, -12, 12), np.clip(p_p.imag, -12, 12)

    if len(d_p) > 0:
        dist_dp = np.abs(d_p - p_p.T)
        acc_dp = np.zeros(d_p.shape, dtype=np.complex128)
        for d_idx in range(len(d_p)):
            idx = np.argsort(dist_dp[d_idx])[:K_nearest]
            f_attract = (np.mean(p_p[idx]) - d_p[d_idx]) * G_dummy_attraction
            f_smell = -gradCost(d_p[d_idx], poles, D) * G_dummy_smell
            acc_dp[d_idx] = f_attract + f_smell
        v_dp = (v_dp + acc_dp * h_physics) * np.exp(-h_physics * kf_dummy)
        d_p = d_p + (v_dp * h_physics)
        
        to_del_p = []
        for d_idx in range(len(d_p)):
            cand = d_p[d_idx][0]
            if np.abs(D(cand)) < 0.2:
                if not any(np.abs(cand - pf) < 0.6 for pf in poles):
                    poles.append(cand)
                    print(f"Iter {i}: Pole Scout {d_idx} found {cand:.4f}")
                    to_del_p.append(d_idx)
                    v_p += np.random.normal(0, 5, p_p.shape) + 1j*np.random.normal(0, 5, p_p.shape)
        if to_del_p:
            d_p = np.delete(d_p, to_del_p, axis=0)
            v_dp = np.delete(v_dp, to_del_p, axis=0)

    g_pz = gradCost(p_z, zeros, N)
    v_z = (v_z + (-g_pz * kg) * h_physics) * np.exp(-h_physics * kf)
    p_z = p_z + (v_z * h_physics) + (g_pz * nudge_factor)
    p_z.real, p_z.imag = np.clip(p_z.real, -12, 12), np.clip(p_z.imag, -12, 12)

    if len(d_z) > 0:
        dist_dz = np.abs(d_z - p_z.T)
        acc_dz = np.zeros(d_z.shape, dtype=np.complex128)
        for d_idx in range(len(d_z)):
            idx = np.argsort(dist_dz[d_idx])[:K_nearest]
            f_attract = (np.mean(p_z[idx]) - d_z[d_idx]) * G_dummy_attraction
            f_smell = -gradCost(d_z[d_idx], zeros, N) * G_dummy_smell
            acc_dz[d_idx] = f_attract + f_smell
        v_dz = (v_dz + acc_dz * h_physics) * np.exp(-h_physics * kf_dummy)
        d_z = d_z + (v_dz * h_physics)
        
        to_del_z = []
        for d_idx in range(len(d_z)):
            cand = d_z[d_idx][0]
            if np.abs(N(cand)) < 0.2:
                if not any(np.abs(cand - zf) < 0.6 for zf in zeros):
                    zeros.append(cand)
                    print(f"Iter {i}: Zero Scout {d_idx} found {cand:.4f}")
                    to_del_z.append(d_idx)
                    v_z += np.random.normal(0, 5, p_z.shape) + 1j*np.random.normal(0, 5, p_z.shape)
        if to_del_z:
            d_z = np.delete(d_z, to_del_z, axis=0)
            v_dz = np.delete(v_dz, to_del_z, axis=0)

    if len(poles) >= 5 and len(zeros) >= 1:
        break
    i += 1
def dupes(root, func, tol=1e-3):
    dupe = 1
    curFunc = func
    while True:
        tVal = root + 1e-6 
        rVal = curFunc(tVal) / (tVal - root)
        if abs(rVal) < tol:
            dupe += 1
            tempFunc = curFunc # scope capture
            curFunc = lambda s: tempFunc(s) / (s - root)
        else:
            break
    return dupe
unique_poles = []
for p in poles:
    if not any(abs(p - up) < 0.1 for up in unique_poles):
        unique_poles.append(p)

unique_zeros = []
for z in zeros:
    if not any(abs(z - uz) < 0.1 for uz in unique_zeros):
        unique_zeros.append(z)

final_poles = []
final_zeros = []

print("\n" + "="*30)
print("   S-PLANE SYSTEM ANALYSIS")
print("="*30)

print("\n--- POLES (Denominator Roots) ---")
for root in unique_poles:
    count = dupes(root, D)
    print(f"Location: {root.real:+.4f} {root.imag:+.4f}j | Multiplicity: {count}")
    for _ in range(count):
        final_poles.append(root)

print("\n--- ZEROS (Numerator Roots) ---")
for root in unique_zeros:
    count = dupes(root, N)
    print(f"Location: {root.real:+.4f} {root.imag:+.4f}j | Multiplicity: {count}")
    for _ in range(count):
        final_zeros.append(root)

print("\n" + "="*30)
print(f"Summary: {len(final_poles)} Poles, {len(final_zeros)} Zeros found.")
print("="*30)