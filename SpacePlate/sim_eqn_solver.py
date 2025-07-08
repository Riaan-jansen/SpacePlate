'''
symoblic python for solving coefficients in Murray thesis SF
'''

import sympy as sym

# make sure you have all the symbols here
# A1, A2, Q0, alpha, a, e = sym.symbols('A1 A2 Q0 alpha a e')

# set up the sympy matrix for 2x2 solver
# A = sym.Matrix([
#     [(-alpha - a**2), (alpha - a**2)],
#     [(alpha - a**2)*e, (-alpha - a**2)*e**-1]
#     ])

# b = sym.Matrix([
#     -2*Q0, 0])

# solution = sym.linsolve((A,b), (A1,A2))
# print(solution)

A1, A2, B1, B2, C1, C2, QM, QMneg, Q0, Q0neg, e1, e2, e3, km, kz0, k0, R, a, pi, d, T= sym.symbols( \
    "A1 A2 B1 B2 C1 C2 QM QMneg Q0 Q0neg e1 e2 e3 km kz0 k0 R a pi d T")

# pressure continuity equations
A = sym.Matrix([
    [(pi*a**2), (pi*a**2)],
    [pi*a**2*e1, pi*a**2/e1]
    ])

Aa = sym.Matrix([
    (Q0 + R*QM), (B1 + B2)*QM])

solution_a = sym.linsolve((A,Aa), (A1,A2))
for sol in solution_a:
    print(f"A1 = {sol[0]},\nA2 = {sol[1]}")

B = sym.Matrix([
    [QM, QM],
    [(e2*QM), (QM/e2)]
])

Bb = sym.Matrix([
    (pi*(a**2) *(A1*e1 + A2/e1)), (pi*(a**2) * (C1 + C2))])

solution_b = sym.linsolve((B,Bb), (B1,B2))
for sol in solution_b:
    print(f"B1 = {sol[0]},\nB2 = {sol[1]}")

C = sym.Matrix([
    [pi*(a**2), pi*(a**2)],
    [pi*(a**2)*e3, pi*(a**2)/e3]
])

Cc = sym.Matrix([
    (B1*e2 + B2/e2)*QM, T*QM])

solution_c = sym.linsolve((C,Cc), (C1,C2))
for sol in solution_c:
    print(f"C1 = {sol[0]},\nC2 = {sol[1]}")

print("\n============================================\n")
# velocity continuity equations
Av = sym.Matrix([
    [k0*QMneg, -k0*QMneg],
    [k0*e1*QMneg, -k0*QMneg/e1]
])

av = sym.Matrix([
    (kz0*d**2 - R*km*d**2), (km*d**2*(B1 - B2))])

solution_av = sym.linsolve((Av,av), (A1,A2))
for sol in solution_av:
    print(f"A1 = {sol[0]},\nA2 = {sol[1]}")


Bv = sym.Matrix([
    [km*d**2, -km*d**2],
    [km*d**2*e2, -km*(d**2)/e2]
])

bv = sym.Matrix([
    k0*QMneg*(A1*e1 - A2/e1), k0*QMneg*(C1 - C2)])

solution_bv = sym.linsolve((Bv,bv), (B1,B2))
for sol in solution_bv:
    print(f"B1 = {sol[0]},\nB2 = {sol[1]}")


Cv = sym.Matrix([
    [k0*QMneg, -k0*QMneg],
    [k0*QMneg*e3, -k0*QMneg/e3]
])

cv = sym.Matrix([
    km*d**2*(B1*e2 - B2/e2), T*km*d**2])

solution_cv = sym.linsolve((Cv,cv), (C1,C2))
for sol in solution_cv:
    print(f"C1 = {sol[0]},\nC2 = {sol[1]}")


eq1 = sym.Eq(A1, (B1*QM*e1 + B2*QM*e1 - Q0 - QM*R)/(a**2*e1**2*pi - a**2*pi))

eq2 = sym.Eq(A2, (-B1*QM*e1 - B2*QM*e1 + Q0*e1**2 + QM*R*e1**2)/(a**2*e1**2*pi - a**2*pi))

eq3 = sym.Eq(B1, (-A1*a**2*e1**2*pi - A2*a**2*pi + C1*a**2*e1*e2*pi + C2*a**2*e1*e2*pi)/(QM*e1*e2**2 - QM*e1))

eq4 = sym.Eq(B2, (A1*a**2*e1**2*e2**2*pi + A2*a**2*e2**2*pi - C1*a**2*e1*e2*pi - C2*a**2*e1*e2*pi)/(QM*e1*e2**2 - QM*e1))

eq5 = sym.Eq(C1, (-B1*QM*e2**2 - B2*QM + QM*T*e2*e3)/(a**2*e2*e3**2*pi - a**2*e2*pi))

eq6 = sym.Eq(C2, (B1*QM*e2**2*e3**2 + B2*QM*e3**2 - QM*T*e2*e3)/(a**2*e2*e3**2*pi - a**2*e2*pi))

eq7 = sym.Eq(A1, (B1*d**2*e1*km - B2*d**2*e1*km + R*d**2*km - d**2*kz0)/(QMneg*e1**2*k0 - QMneg*k0))

eq8 = sym.Eq(A2, (B1*d**2*e1*km - B2*d**2*e1*km + R*d**2*e1**2*km - d**2*e1**2*kz0)/(QMneg*e1**2*k0 - QMneg*k0))

eq9 = sym.Eq(B1, (-A1*QMneg*e1**2*k0 + A2*QMneg*k0 + C1*QMneg*e1*e2*k0 - C2*QMneg*e1*e2*k0)/(d**2*e1*e2**2*km - d**2*e1*km))

eq10 = sym.Eq(B2, (-A1*QMneg*e1**2*e2**2*k0 + A2*QMneg*e2**2*k0 + C1*QMneg*e1*e2*k0 - C2*QMneg*e1*e2*k0)/(d**2*e1*e2**2*km - d**2*e1*km))

eq11 = sym.Eq(C1, (-B1*d**2*e2**2*km + B2*d**2*km + T*d**2*e2*e3*km)/(QMneg*e2*e3**2*k0 - QMneg*e2*k0))

eq12 = sym.Eq(C2, (-B1*d**2*e2**2*e3**2*km + B2*d**2*e3**2*km + T*d**2*e2*e3*km)/(QMneg*e2*e3**2*k0 - QMneg*e2*k0))

eq_names = [f"eq{i}" for i in range(1,10)]
eqs_1 = [globals()[name] for name in eq_names]

unknowns = [A1, A2, B1, B2, C1, C2, R]

params = {a: 1.2, d: 8, pi: 3.14159}
eqs = [eq.subs(params) for eq in eqs_1]

solution = sym.solve(eqs_1, unknowns, dict=True)
sol = solution[0]

eq_T = eq11

for unknown in unknowns:
    eq_T = eq_T.subs(unknown, sol[unknown])

print(eq_T)
