import numpy as np
import matplotlib.pyplot as plt


def Fp(F0):
    # F0 = 1-3*p/4
    Numer = (F0**2 + ((1-F0)**2)/9) 
    Denom = F0**2 + 2*F0*(1-F0)/3 + 5*((1-F0)**2)/9
    return Numer / Denom

def yd(F0):
    # F0 = 1-3*p/4
    ps = F0**2 + 2*F0*(1-F0)/3 + 5*((1-F0)**2)/9
    return ps / 2

def enhanced_recurrence(F0, r):
    yd = 1
    p00 = F0
    p01 = (1 - F0) / 3
    p10 = p01
    p11 = p01

    # print(p00, p01, p10, p11)
    # print("\n")

    for _ in range(r):
        p_pass = (p00 + p10)**2 + (p01+p11)**2
        q00 = (p00**2 + p10**2) / p_pass 
        q01 = 2 * p00 * p10 / p_pass 
        q10 = (p01**2 + p11**2) / p_pass 
        q11 = 2 * p01 * p11 / p_pass

        # print(q00, q01, q10, q11, p_pass)

        ls = sorted([q01, q10, q11])
        # print(ls)

        p00 = q00
        p01 = ls[1]
        p10 = ls[0]
        p11 = ls[2]

        yd = yd * p_pass / 2

    return p00, yd

if __name__ == "__main__":
    p = 0.2
    F0 = 1 - 3*p/4
    num_iter = 1
    print("recurrence:", enhanced_recurrence(F0, num_iter))

    N = 10
    m0 = 5
    intact = F0**N
    errors = (1-intact)*(2**-m0)

    yH = 1

    fH = intact / (intact + errors)
    yH = yH * (intact + errors) * (1-m0/N)

    r2 = 6
    m2 = 9
    intact = fH**r2
    errors = (1-intact)*(2**-m2)

    fH = intact / (intact + errors)
    yH = yH * (intact + errors) * (1-m2/(r2*(N-m0)))

    print("haar Hamiltonian:", fH, yH)

    
