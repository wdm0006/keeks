Keeks
=====

A python library for bankroll allocation strategies.

Very much a work in progress.

Currently includes a few concepts:

 * Opportunity: an object that represents some bet. It has a an expected proability of paying off, some odds that it
 pays, and maybe an explicitly known outcome to use in validation. If not, an outcome can be simulated easily.
 * Bank: a bank is basically just an object that holds the float of money you have. It supports using only a fraction
 of the total acct on any given set of bets, and a max drawdown limit.
 * Strategy: a strategy is the higher level object that ties it all together. The first strategy implimented is the Kelly
 Criterion. The idea is that given a set of opportunities with odds and expected probabilities of winning, we can produce
 a nice set of bets to make with whatever cash is available in our bank.


Related Projects
================

This is part of a 3 project ecosystem, elote is a python library for creating rating systems to rank things. And
keeks-elote, as you might imagine from the name, is a glue-library that combines the rating systems in elote and the
bankroll strategies in keeks to simulate the performance of a rating system and betting strategy pair on historical data.
