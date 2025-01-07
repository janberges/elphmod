#!/usr/bin/env python3

# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import sympy

# We want to calculate:
#
# 1/V integral d^2k delta[e1(k)] delta[e2(k)] f(k)

def simpler(x):
    return sympy.factor(sympy.expand(x))

A, B, C, a, b, c, X, T, t, l = sympy.symbols('A B C a b c X T t l')

#  X \
#     \   C______________A'
#      \  /\            /
#       \/  \          /
#       /\   \        /
#   a2 /  \   \      /
#     /    \   \    /
#    /      \   \  /
#   /________\___\/
#  A      a1  \   B
#              \
#   (X - A) / (B - A) a1 T + (X - A) / (C - A) a2 (1 - T)
# = (X - a) / (b - a) a1 t + (X - a) / (c - a) a2 (1 - t)

K1 = (X - A) / (B - A) * T
k1 = (X - a) / (b - a) * t
K2 = (X - A) / (C - A) * (1 - T)
k2 = (X - a) / (c - a) * (1 - t)

solution = sympy.solve({K1 - k1, K2 - k2}, {T, t})

for key in solution:
    solution[key] = simpler(solution[key])

K1 = simpler(K1.subs(solution))
k1 = simpler(k1.subs(solution))
K2 = simpler(K2.subs(solution))
k2 = simpler(k2.subs(solution))

assert K1 == k1
assert K2 == k2

K3 = simpler(1 - K1 - K2)

# Integral over region of intersection at e = E = 0:
#
#   integral dkx dky delta[grad(e1) k] delta[grad(e2) k]
# = integral dk1 dk2 / |det[d(k1, k2) / d(kx, ky)]| delta(k1) delta(k2)
#
# with dk1,2/dk = grad(e1,2)

V = l ** 2 * sympy.sqrt(3) / 2 # BZ area

grad1x = (B - A) / l
grad1y = (C - (A + B) / 2) / (sympy.sqrt(3) / 2 * l)
grad2x = (b - a) / l
grad2y = (c - (a + b) / 2) / (sympy.sqrt(3) / 2 * l)

det = simpler(grad1x * grad2y - grad1y * grad2x)

weight = simpler(1 / abs(det * V))

print(K1)
print(K2)
print(K3)
print(weight)
