---
eip: 607
title: "Hardfork Meta: Spurious Dragon"
author: Alex Beregszaszi (@axic)
type: Meta
status: Final
created: 2017-04-23
requires: 155, 160, 161, 170, 608
legacy link: https://eips.ethereum.org/EIPS/eip-607
---

## Abstract

This specifies the changes included in the hard fork named Spurious Dragon.

## Specification

- Codename: Spurious Dragon
- Aliases: State-clearing
- Activation:
  - Block >= 2,675,000 on Mainnet
  - Block >= 1,885,000 on Morden
- Included EIPs:
  - [EIP-155](https://eips.ethereum.org/EIPS/eip-155) (Simple replay attack protection)
  - [EIP-160](https://eips.ethereum.org/EIPS/eip-160) (EXP cost increase)
  - [EIP-161](https://eips.ethereum.org/EIPS/eip-161) (State trie clearing)
  - [EIP-170](https://eips.ethereum.org/EIPS/eip-170) (Contract code size limit)

## References

1. https://blog.ethereum.org/2016/11/18/hard-fork-no-4-spurious-dragon/

## Copyright

Copyright and related rights waived via [CC0](https://creativecommons.org/publicdomain/zero/1.0/).