---
eip: 609
title: "Hardfork Meta: Byzantium"
author: Alex Beregszaszi (@axic)
type: Meta
status: Final
created: 2017-04-23
requires: 100, 140, 196, 197, 198, 211, 214, 607, 649, 658
legacy link: https://eips.ethereum.org/EIPS/eip-609
---

## Abstract

This specifies the changes included in the hard fork named Byzantium.

## Specification

- Codename: Byzantium
- Aliases: Metropolis/Byzantium, Metropolis part 1
- Activation:
  - Block >= 4,370,000 on Mainnet
  - Block >= 1,700,000 on Ropsten testnet
- Included EIPs:
  - [EIP-100](https://eips.ethereum.org/EIPS/eip-100) (Change difficulty adjustment to target mean block time including uncles)
  - [EIP-140](https://eips.ethereum.org/EIPS/eip-140) (REVERT instruction in the Ethereum Virtual Machine)
  - [EIP-196](https://eips.ethereum.org/EIPS/eip-196) (Precompiled contracts for addition and scalar multiplication on the elliptic curve alt_bn128)
  - [EIP-197](https://eips.ethereum.org/EIPS/eip-197) (Precompiled contracts for optimal ate pairing check on the elliptic curve alt_bn128)
  - [EIP-198](https://eips.ethereum.org/EIPS/eip-198) (Precompiled contract for bigint modular exponentiation)
  - [EIP-211](https://eips.ethereum.org/EIPS/eip-211) (New opcodes: RETURNDATASIZE and RETURNDATACOPY)
  - [EIP-214](https://eips.ethereum.org/EIPS/eip-214) (New opcode STATICCALL)
  - [EIP-649](https://eips.ethereum.org/EIPS/eip-649) (Difficulty Bomb Delay and Block Reward Reduction)
  - [EIP-658](https://eips.ethereum.org/EIPS/eip-658) (Embedding transaction status code in receipts)

## References

1. https://blog.ethereum.org/2017/10/12/byzantium-hf-announcement/

## Copyright

Copyright and related rights waived via [CC0](https://creativecommons.org/publicdomain/zero/1.0/).