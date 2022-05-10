import langdetect as l
from googletrans import Translator


PATTERN = [3, 4, 5, 6, 7, 8, 14, 15]

PERM_KEYS_56 = [57, 49, 41, 33, 25, 17, 9,
                1, 58, 50, 42, 34, 26, 18,
                10, 2, 59, 51, 43, 35, 27,
                19, 11, 3, 60, 52, 44, 36,
                63, 55, 47, 39, 31, 23, 15,
                7, 62, 54, 46, 38, 30, 22,
                14, 6, 61, 53, 45, 37, 29,
                21, 13, 5, 28, 20, 12, 4]

PERM_KEYS_48 = [14, 17, 11, 24,  1, 5,
                3, 28, 15,  6, 21, 10,
                23, 19, 12,  4, 26, 8,
                16,  7, 27, 20, 13, 2,
                41, 52, 31, 37, 47, 55,
                30, 40, 51, 45, 33, 48,
                44, 49, 39, 56, 34, 53,
                46, 42, 50, 36, 29, 32]

MSG_IP = [58, 50, 42, 34, 26,  18, 10, 2,
          60, 52, 44, 36, 28, 20, 12, 4,
          62, 54, 46, 38, 30, 22, 14, 6,
          64, 56, 48, 40, 32, 24, 16, 8,
          57, 49, 41, 33, 25, 17, 9, 1,
          59, 51, 43, 35, 27, 19, 11, 3,
          61, 53, 45, 37, 29, 21, 13, 5,
          63, 55, 47, 39, 31, 23, 15, 7]

E_SELECT = [32, 1, 2, 3, 4, 5,
            4, 5, 6, 7, 8, 9,
            8, 9, 10, 11, 12, 13,
            12, 13, 14, 15, 16, 17,
            16, 17, 18, 19, 20, 21,
            20, 21, 22, 23, 24, 25,
            24, 25, 26, 27, 28, 29,
            28, 29, 30, 31, 32, 1]

S_TABLES = [[[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
             [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
             [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
             [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]],

            [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
             [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
             [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
             [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]],

            [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
             [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
             [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
             [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]],

            [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
             [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
             [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
             [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]],

            [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
             [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
             [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
             [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]],

            [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
             [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
             [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
             [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]],

            [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
             [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
             [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
             [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]],

            [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
             [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
             [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
             [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]]]

MSG_FP = [40, 8, 48, 16, 56, 24, 64, 32,
          39, 7, 47, 15, 55, 23, 63, 31,
          38, 6, 46, 14, 54, 22, 62, 30,
          37, 5, 45, 13, 53, 21, 61, 29,
          36, 4, 44, 12, 52, 20, 60, 28,
          35, 3, 43, 11, 51, 19, 59, 27,
          34, 2, 42, 10, 50, 18, 58, 26,
          33, 1, 41, 9, 49, 17, 57, 25]

F_PERM = [16, 7, 20, 21,
          29, 12, 28, 17,
          1, 15, 23, 26,
          5, 18, 31, 10,
          2, 8, 24, 14,
          32, 27, 3, 9,
          19, 13, 30, 6,
          22, 11, 4, 25]

LEFT_SHIFTS = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]


def permute(sequence, table, size):
    """
    This function rearranges the bits of a given sequence based on a provided table for DES
    :param sequence: The bit sequence to rearrange
    :param table: The table to use to map bits to their new location in the sequence
    :param size: The size of the original sequence in bits
    :return: The newly rearranged bit sequence
    """
    sequence_new = 0x0
    for b in table:
        bit = (sequence >> (size - b)) & 0b1  # Gets the new bit based on perm table
        sequence_new = (sequence_new | bit) << 1  # Adds the new bit to the end of the new key
    return sequence_new >> 1  # Removes trailing 0


def generate_16_sub_keys(key):
    """
    Given an initial key, this function generates 16 sub keys, which each one will be used at a different cycle
    in the DES encryption process
    :param key: The original key
    :return: A list of 16 subkeys generated from the initial key
    """
    sub_keys = []

    # Generate C0 (right half) and D0 (left half)
    kn = permute(key, PERM_KEYS_56, 64)
    cn, dn = (kn >> 28) & 0xFFFFFFF, kn & 0xFFFFFFF  # Splits the key

    # Generates C1/D1 through C16/D16
    for i in range(16):
        for s in range(LEFT_SHIFTS[i]):
            # Bit magic to shift the keys, wrapping the overflowed bit back to the end
            cn = ((cn & 0x8000000) >> 27) | ((cn << 1) & 0xFFFFFFF)
            dn = ((dn & 0x8000000) >> 27) | ((dn << 1) & 0xFFFFFFF)

        # Apply second permutation on the concatenated sub keys
        kn = permute((cn << 28) | dn, PERM_KEYS_48, 56)
        sub_keys.append(kn)

    return sub_keys


def f(rn, kn):
    """
    This function generates a confused and diffused sequence of the right 32 bits for the given cycle
    :param rn: The right have have the scrambled, encoded block
    :param kn: The key for the current cycle we on
    :return: The new right half of the encoded block for a given cyce
    """
    e_rn = permute(rn, E_SELECT, 32)
    bn = e_rn ^ kn

    # Generates the 8 b subsets, each one consisting of 6 bits
    rn_new = 0x00000000
    for i in range(7, -1, -1):
        sub_b = bn >> (6 * i) & 0b111111

        # Use that S table to map 6 bits to new 4 bit sequence
        si_table = S_TABLES[7-i]
        row = ((sub_b >> 5) << 1) | (sub_b & 0b1)
        col = (sub_b & 0b011110) >> 1
        rn_new = (rn_new | si_table[row][col]) << 4

    rn_new >>= 4
    return permute(rn_new, F_PERM, 32)


def encode_message(block, sub_keys):
    """
    This function is responsible for encoding a single 64 bit message according to DES
    :param block: The 64 bit block to encode
    :param sub_keys: All the sub keys generated from the original key
    :return: The encrypted 64-bit block
    """
    block_n = permute(block, MSG_IP, 64)
    ln, rn = (block_n >> 32) & 0xFFFFFFFF, block_n & 0xFFFFFFFF  # Splits the key

    # Runs through each individual encoding cycle
    for cyc in range(16):
        ln_old = ln
        ln = rn
        rn = ln_old ^ f(rn, sub_keys[cyc])

    new_block = (rn << 32) | ln
    return permute(new_block, MSG_FP, 64)


def encrypt_des(plaintext, key):
    """
    This function takes in a plaintext and encrypts it according to DES
    :param plaintext: The text to encrypt using DES
    :param key: The original key
    :return: The new cipher text
    """
    msg_bin = int.from_bytes(bytes(plaintext.encode("ascii")), byteorder="big")
    bit_len = msg_bin.bit_length()
    bit_len = bit_len + 8 - (bit_len % 8)
    encoded_msg = ""

    # Does the message need padding
    leftover_bits = bit_len % 64
    padding = 64 - leftover_bits if leftover_bits else 0
    msg_bin <<= padding
    bit_len += padding

    # Break the message into blocks and encrypt it
    sub_keys = generate_16_sub_keys(key)
    for i in range(bit_len // 64 - 1, -1, -1):
        block = msg_bin >> (i * 64) & 0xFFFFFFFFFFFFFFFF
        encoded_block = encode_message(block, sub_keys)
        encoded_msg += format(encoded_block, "016x")

    return encoded_msg


def decrypt_des(cipher, key, length):
    """
    So we got a cipher and a key, so why don't we try to decrypt it?
    :param length: The length of the cipher text
    :param cipher: The encrypted message
    :param key: The key we'll use to try to decrypted
    :return: An attempt at decrypting the message using the provided key
    """
    decoded_msg = ""

    # Break the message into blocks and decrypts it
    sub_keys = [k for k in reversed(generate_16_sub_keys(key))]
    for i in range(length // 16 - 1, -1, -1):
        block = cipher >> (i * 64) & 0xFFFFFFFFFFFFFFFF
        decoded_block = encode_message(block, sub_keys)

        decode_bytes = decoded_block.to_bytes(16, byteorder="big")
        for b in decode_bytes:
            if b >= 128:
                return
            decoded_msg += chr(b)

    return decoded_msg


def brute_force(cipher, length):
    """
    Given a cipher and a subset of keys, test all possible options to see what key decrypts the cipher
    :param length: The length of the cipher
    :param cipher: The encrypted cipher text
    :return:
    """

    possible_plain = []
    detector = Translator()
    key_base = 0x0
    s, e = 0, 16
    hex_num = 1

    # Loops through all possible keys
    while key_base <= 0xFFFFF:

        # Is the hex sequence comprised of valid values
        valid = True
        for i in range(hex_num):
            if (key_base >> (4 * i)) & 0xF not in PATTERN:
                valid = False
                break

        if valid:
            key = (key_base << 44) + 0x76f30303030
            attempt = decrypt_des(cipher, key, length)

            # Is the decrypted message in english?
            if attempt:
                top = detector.detect(attempt).lang
                if top == "en":
                    possible_plain.append((key, attempt))

        key_base += 1
        s += 1

        # New hex
        if s >= e:
            s = 0
            e *= 16
            hex_num += 1

    return possible_plain


def main():
    detector = Translator()
    print(detector.detect('boba tea'))
    while True:
        cmd = input('''Type 'e' to encrypt\nType 'd' to decrypt\nType 'b' to brute force\nType 'q' to quit\n\nOption: ''').lower()

        # Let's encrypt
        if cmd == 'e':
            plaintext = input("Enter a plaintext to encrypt: ")
            key = int(input("Enter a key: "), 16)
            cipher = encrypt_des(plaintext, key)
            print("The Cipher Text:", cipher.upper(), end="\n\n")

        # Let's decrypt
        elif cmd == 'd':
            cipher = input("Enter a cipher to decrypt: ")
            key = int(input("Enter a key: "), 16)
            cipher_bin = int(cipher, 16)
            length = len(cipher)
            decoded = decrypt_des(cipher_bin, key, length)
            print("The Decrypted Plaintext:", decoded, end="\n\n")

        # Let's brute force
        elif cmd == 'b':
            cipher = input("Enter a cipher to decrypt: ")
            cipher_bin = int(cipher, 16)
            length = len(cipher)
            possible_plain = brute_force(cipher_bin, length)

            for p in possible_plain:
                print(format(p[0], "016x") + ": ", p[1])

        # Let's quit
        elif cmd == 'q':
            break

        # Invalid command
        else:
            print("Invalid command '{}'".format(cmd), end="\n\n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
