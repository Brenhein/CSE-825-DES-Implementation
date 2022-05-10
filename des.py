cipher = [3185, 2038, 2460, 2550]
p, q, e = 53, 61, 17
n, o_n = p * q, (p - 1) * (q - 1)

for c in cipher:
    m = 0
    while True:
        if (m ** e) % n == c:  # Does the potential plaintext encrypted equal cipher
            print(m, end=" ")
            break
        m += 1


