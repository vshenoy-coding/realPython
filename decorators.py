#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 03:27:03 2022

@author: vms
"""
# Using decorator and wrapper functions. Uncomment decorated function lines
# and print statements to see results.
##non pie syntax
# def my_decorator(func):
#     def wrapper():
#         print("Something is happening before the function is called.")
#         func()
#         print("Something is happening after the function is called.")
#     return wrapper

# def intermediate():
#     print("This is what is happening in the intermediate stage.")

# intermediate = my_decorator(intermediate)
# intermediate()

#pie syntax
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def intermediate():
    print("This is what is happening in the intermediate stage.")

# intermediate()

###
from datetime import datetime

##no pie syntax
# def not_during_the_night():
#     def wrapper(*args):
#         "What hour of the day is it?"
#         print(datetime.now().hour)
#         if 7 <= datetime.now().hour < 22:
#             func()
#         else:
#             print("It's nighttime.")
#     return wrapper

# def day():
#     print("It is still not night.")
 

# day = not_during_the_night()
# day()

###
#pie syntax   
def not_during_the_night(func):
    def wrapper(*args):
        "What hour of the day is it?"
        print(datetime.now().hour)
        if 7 <= datetime.now().hour < 22:
            func()
        else:
            print("It's nighttime.")
    return wrapper
 

@not_during_the_night
def day():
    print("It is still not night.")

# day()

###
def do_twice(func):
    def wrapper_do_twice(*args, **kwargs):
        func(*args, **kwargs)
        #func(*args, **kwargs)
    return wrapper_do_twice

#from decorators import do_twice

#@do_twice
# def twice():
#     #print("2")

@do_twice
def greet(name=None):
    name = " " if name is None else name
    #print(f"Hello, {name}")

# greet("World")
###
import functools

def again_twice(func):
      @functools.wraps(func) 
      # so that return_greeting function is not confused as wrapper_again_twice
      def wrapper_again_twice(*args, **kwargs):
          func(*args, **kwargs)
          return func(*args, **kwargs)
      return wrapper_again_twice

#from decorators import again_twice
@again_twice
def return_greeting(name=None):
    name = " " if name is None else name
    print("Creating greeting")
    return f"Hi {name}"

# hi_steve = return_greeting("steve")
# print(hi_steve)

###
#basic decorator function template
def decorator(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        value = func(*args, **kwargs)
        # Do something after
        return value
    return wrapper_decorator

###

import time

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

@timer
def waste_some_time(num_times):
    for _ in range(num_times):
        sum([i**2 for i in range(10000)])


# waste_some_time(1)
# waste_some_time(999)
# waste_some_time(1000)

###
# def debug(func):
#     """Print the function signature and return value"""
#     @functools.wraps(func)
#     def wrapper_debug(*args, **kwargs):
#         args_repr = [repr(a) for a in args]                      # 1
#         kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
#         signature = ", ".join(args_repr + kwargs_repr)           # 3
#         print(f"Calling {func.__name__}({signature})")
#         value = func(*args, **kwargs)
#         print(f"{func.__name__!r} returned {value!r}")           # 4
#         return value
#     return wrapper_debug

# 1 Create a list of the positional arguments. Use repr() to get 
# string representing each argument.

# 2 Create a list of the keyword arguments. The f-string formats each argument 
# as key=value where the !r specifier means that repr() is used to represent 
# the value.

# 3 The lists of positional and keyword arguments is joined together to one 
# signature string with each argument separated by a comma.

# The return value is printed after the function is executed.

# @debug
# def make_greeting(name=None, age=None):
#     name = " " if name is None else name 
#     if age is None:
#         return f"Hi {name}!"
#     else:
#         return f"Whoa {name}! {age} already, you are growing up!"

# make_greeting()
# print("\n")
# make_greeting("Benjamin")
# print("\n")
# make_greeting("Richard", age=112)
# print("\n")
# make_greeting("Tyler", age=25)

###

def debug(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")           # 4
        return value
    return wrapper_debug
import math
#from decorators import debug

# Apply a decorator to a standard library function
math.factorial = debug(math.factorial)

def approximate_e(terms=18):
    return sum(1 / math.factorial(n) for n in range(terms))

# print(approximate_e(15))
###

def slow_down(func):
    """Sleep 1 second before calling the function"""
    @functools.wraps(func)
    def wrapper_slow_down(*args, **kwargs):
        start_time = time.time()
        time.sleep(1)
        end_time = time.time()
        diff_time = end_time - start_time
        print("Time elapsed is", diff_time)
        return func(*args, **kwargs)
    return wrapper_slow_down

@slow_down
def countdown(from_number):
    if from_number < 1:
        print("Liftoff!")
    else:
        print(from_number)
        countdown(from_number - 1)

###
import random
PLUGINS = dict()

def register(func):
    """Register a function as a plug-in"""
    PLUGINS[func.__name__] = func
    return func

@register
def say_hello(name):
    return f"Hello {name}"

@register
def be_awesome(name):
    return f"Yo {name}, together we are the awesomest!"

def randomly_greet(name):
    greeter, greeter_func = random.choice(list(PLUGINS.items()))
    print(f"Using {greeter!r}")
    return greeter_func(name)


