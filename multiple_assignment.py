#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 02:23:06 2022

@author: vms
"""

items = [1, 2, 3, 4, 2, 1]

# show that item is of type list
print(type(items))

# enumerate list items, starting at index 1, then print out count and value
# in enumerated list 
a = list(enumerate(items, start=1))
print(a)
for count, value in a:
    print(count, value)

# use zip and enumerate methods to create enumerated list of tuples
b = list(enumerate(zip(items, reversed(items))))
print(b)

# for loop to raise ValueError if first value in nested tuple doesn't match 
# second value in nested tuple
# for i, (first, last) in enumerate(zip(items, reversed(items))):
#     if first != last:
#         raise ValueError(f"Item {i} doesn't match: {first} != {last}")


numbers = [i*2 for i in range(1,15)]
[first, *rest] = numbers

# demonstrate how * command can be used as a slicing method
print(first)
print(rest)

# print out elements in colors list using a for loop in range of the length
# the color list
colors = ["red", "green", "blue", "purple"]
for i in range(len(colors)):
    print(colors[i])

# print out order of presidents and their names using range function in for loop
# presidents = ["Washington", "Adams", "Jefferson", "Madison", "Monroe", "Adams", "Jackson"]
# for i in range(len(presidents)):
#     print("President {}: {}".format(i + 1, presidents[i]))


# print out order of presidents and their names, using enumerate function in for loop
presidents = ["Washington", "Adams", "Jefferson", "Madison", "Monroe", "Adams", "Jackson"]
for num, name in enumerate(presidents, start=1):
    print("President {}: {}".format(num, name))
    
# define list of colors and decimal values of color ratios
colors = ["red", "green", "blue", "purple"]
ratios = [0.2, 0.3, 0.1, 0.4]

# print out percentage ratios and color using a two iterables in a for loop
# print(list(enumerate(colors)))
# for i, color in enumerate(colors):
#     ratio = ratios[i]
#     print("{}% {}".format(ratio * 100, color))

# create list of tuples of iterables colors and ratios
print(list(zip(colors, ratios)))
    
# print out percentage ratios and color using two iterables in a for loop,
# but with the zip method rather than indexing the ratios
for color, ratio in zip(colors, ratios):
    print("{}% {}".format(ratio * 100, color))
    
# define lists of pallete hues
palletes = ["pink", "lime", "aqua", "indigo"]

# print out percentage ratios and color-hue using three iterables in a for loop,
# through using the zip method to loop through three lists of the same length
for color, ratio, pallete in zip(colors, ratios, palletes):
    print("Percentage: {}%, Hue: {}-{}".format(ratio * 100, color, pallete))

# Sample code for aggregating elements from each of the iterables, given that
# iterables are not of the same length, so missing values are populated with fillvalue. 
# Iteration in zip_longest function continues until the longest iterable is exhausted.
# Source of code: https://docs.python.org/3/library/itertools.html#itertools.zip_longest
# def zip_longest(*args, fillvalue=None):

#     iterators = [iter(it) for it in args]
#     num_active = len(iterators)
#     if not num_active:
#         return
#     while True:
#         values = []
#         for i, it in enumerate(iterators):
#             try:
#                 value = next(it)
#             except StopIteration:
#                 num_active -= 1
#                 if not num_active:
#                     return
#                 iterators[i] = repeat(fillvalue)
#                 value = fillvalue
#             values.append(value)
#         yield tuple(values)

# a = zip_longest('ABCD', 'xy', fillvalue='-') #--> Ax By C- D-
# print(list(enumerate(a)))

# import random module and define function main to create a guessing game,
# where a number is randomly generated, Hot and Cold are 10 greater than and less
# than, respectively, the number, and the number of guesses taken is stored,
# which keeps track of the number of guesses.
# if and elif statements
# keep track of whether the guess is too low, too high, or getting closer
# to the actual value (provided the guess is within 10 units of the actual number).
# The inner while loop continues until the correct number is input, and
# the program tells you how many guesses it took to guess the right number
import random
def main():
      guessesTaken = 0
      guess = 0
      number = random.randint (1, 1000)
      print('Guess a number from 1 to 1000. debug = ',number,'\n')
      guess = float(input('  '))
      Hot  = number + 10
      Cold = number - 10
      diff = guess - number
      while guess != number:
          # update value of diff
          diff = guess - number
          guessesTaken = guessesTaken + 1
          print(diff)
          if guess > Hot:
              print('Too High!')
          elif guess < Cold:
            print('Your guess is too low.')
          elif diff > 0 and guess <= Hot:
            print('Getting warmer, but your guess is still too high.')
          elif diff < 0 and guess >= Cold:
            print('Getting warmer, but your guess is still too low.')
          # else:
          #   print('Error')
          guess = float(input('  '))
   
      if guess == number:
          print('You got it! You guessed the number in ' + str(guessesTaken) + ' guesses!')

main()
# create a guessing game, where a number is randomly generated, diff is the
# difference between the number and the guess, and if and elif statements
# keep track of whether the guess is too low, too high, or getting closer
# to the actual value (provided the guess is within 10 units of the actual number)
# import random
# def main():
#      guessesTaken = 0
#      guess = 0
#      number = random.randint (1, 1000)
#      print('Guess a number from 1 to 1000. debug = ',number,'\n')
#      guess = float(input('  '))

#      while guess != number:
#         # update values of diff
#          diff = guess - number
#          guessesTaken = guessesTaken + 1
#          print(diff)
#          if diff > 10:
#              print('Too High!')
#          elif diff < -10:
#             print('Your guess is too low.')
#          elif 0 < diff <= 10:
#              print('Getting warmer, but your guess is still too high.')
#          elif -10 <= diff < 0:
#              print('Getting warmer, but your guess is still too low.')
#          # else:
#          #     print('Error')
#          guess = float(input('  '))
   
#      if guess == number:
#          print('You got it! You guessed the number in ' + str(guessesTaken) + ' guesses!')

# main()


