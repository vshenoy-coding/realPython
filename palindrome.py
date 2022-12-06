#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 01:46:20 2022

@author: vms
"""
# define function infinite_sequence to create a generator function that is an infinite
# sequence
def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1
gen = infinite_sequence() 

# print next element of generator
#print(next(gen))

# define function is_palindrome to find palindromes as numbers are iterated through,
# bypassing numbers iterated through that are less than 10
def is_palindrome(num):
    # Skip single-digit inputs
    if num // 10 == 0:
        return False
    temp = num
    reversed_num = 0

    while temp != 0:
        reversed_num = (reversed_num * 10) + (temp % 10)
        temp = temp // 10

    if num == reversed_num:
        return True
    else:
        return False

# print out list of palindromes
# to restrict number of palindromes, change while True: statement in 
# def infinite_sequence to something like while < num 2000
# for i in infinite_sequence():
#       pal = is_palindrome(i)
#       if pal:
#           print(i)

# define function infinite_palindrome to iterate through palindromes up to 
# infinity
def infinite_palindromes():
    num = 0

    while True:
        if is_palindrome(num):
            i = (yield num)
            if i is not None:
                num = i

        num += 1
# create coroutine, or a generator function into which you can pass data.   
# pal_gen = infinite_palindromes()
# for i in pal_gen:
#     digits = len(str(i))
#     print(i)
#     pal_gen.send(10 ** (digits))

# throw ValueError if palindrome has more than 5 digits
#pal_gen = infinite_palindromes()
# for i in pal_gen:

#     print(i)

#     digits = len(str(i))

#     if digits == 5:

#         pal_gen.throw(ValueError("We don't like large palindromes"))

#     pal_gen.send(10 ** (digits))
    
# StopIteration when number of digits in palindrome reaches 5 
pal_gen = infinite_palindromes()

for i in pal_gen:

    print(i)

    digits = len(str(i))

    if digits == 5:

        pal_gen.close()

    pal_gen.send(10 ** (digits))