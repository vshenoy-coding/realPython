#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 06:57:45 2022

@author: vms
"""

import functools
import time

def slow_down(_func=None, *, rate=1):
    # Sleep given amount of seconds before calling the function
    def decorator_slow_down(func):
        @functools.wraps(func)
        def wrapper_slow_down(*args, **kwargs):
            #start_time = time.time()
            time.sleep(rate)
            #end_time = time.time()
            #diff_time = end_time - start_time
            #print("Time elapsed is:", diff_time)
            return func(*args, **kwargs)
        return wrapper_slow_down
    
    if _func is None:
        return decorator_slow_down
    
    else:
        return decorator_slow_down(_func)
    
@slow_down(rate=2)
def countdown(from_number):
    if from_number < 1:
        print("Liftoff!")
        
        
    else:
        print(from_number)
        countdown(from_number - 1)
        
###
#Creating singletons and using them on classes
# parameter is cls rather than func since we are using a decorator on a class
# and not a function

def singleton(cls):
    # Make a class a Singleton class (only one instance)
    @functools.wraps(cls)
    def wrapper_singleton(*args, **kwargs):
        if not wrapper_singleton.instance:
            wrapper_singleton.instance = cls(*args, **kwargs)
        return wrapper_singleton.instance
    wrapper_singleton.instance = None
    return wrapper_singleton

@singleton
class TheOne:
    pass

first_one = TheOne()
another_one = TheOne()

#print(id(first_one))
#print(id(another_one))
#ids should be the same, since first_one is the exact same instance as another_one

#print(first_one is another_one)
#boolean should be True, since first_one is the exact same instance as another_one

###
# Caching Return Values: recursvie definition of fibonacci series

def count_calls(func):
    @functools.wraps(func)
    def wrapper_count_calls(*args, **kwargs):
        wrapper_count_calls.num_calls += 1
        print(f"Call {wrapper_count_calls.num_calls} of {func.__name__!r}")
        return func(*args, **kwargs)
    wrapper_count_calls.num_calls = 0
    return wrapper_count_calls

#from decorators import count_calls

@count_calls
def fibonacci(num):
    if num < 2:
        return num
    return fibonacci(num - 1) + fibonacci(num - 2)

#print(fibonacci(20))
#print(fibonacci.num_calls)

# problem: code keeps recalculating Fibonacci numbers that are already known.

# caching of fibonacci calculations is needed

def cache(func):
    # Keep a cache of previous function calls
    @functools.wraps(func)
    def wrapper_cache(*args, **kwargs):
        cache_key = args + tuple(kwargs.items())
        if cache_key not in wrapper_cache.cache:
            wrapper_cache.cache[cache_key] = func(*args, **kwargs)
        return wrapper_cache.cache[cache_key]
    wrapper_cache.cache = dict()
    return wrapper_cache

@cache
@count_calls
def fibonacci(num):
    if num < 2:
        return num
    return fibonacci(num - 1) + fibonacci(num - 2)

# use @functools.lru_cache for Least Recently Used (LRU) cache
# instead of writing your own cache decorator

@functools.lru_cache(maxsize=4) 
# maxsize parameter specifies how many recent calls are cached, 
# default value is 128 by can be set to None to cache all function calls -
# however this may cause memory problems when caching many large objects
def fibonacci(num):
    print(f"Calculating fibonacci({num})")
    if num < 2:
        return num
    return fibonacci(num - 1) + fibonacci(num - 2)

###
# Adding information about units

def set_unit(unit):
    # Register a unit on a function
    def decorator_set_unit(func):
        func.unit = unit
        return func
    return decorator_set_unit
        
import math

@set_unit("cm^3")
def volume(radius, height):
    if radius<=0 or height<=0:
        raise ValueError("negative values can't be entered")
    return math.pi * radius**2 * height

# import pint module for assistance with unit conversion
import pint

ureg = pint.UnitRegistry()
vol = volume(3,5) * ureg(volume.unit)

# print volume vol in cubic centimeters
print(vol)

# convert volume from cubic centimeters to cubic inches, and 
# print out value with units
print(vol.to("cubic inches"))
# convert volume from cubic centimeters to gallons in magnitude, and
# print out value without units      
print(vol.to("gallons").m)

# modify the decorator to return a pint Quantity directly. 
# Such a Quantity is made by multiplying a value with the unit. 
# In pint, units must be looked up in a UnitRegistry. 
# The registry is stored as a function attribute to avoid cluttering the namespace:

def use_unit(unit):
    # Have a function return a Quantity with given unit
    use_unit.ureg = pint.UnitRegistry()
    def decorator_use_unit(func):
        @functools.wraps(func)
        def wrapper_user_unit(*args, **kwargs):
            value = func(*args, **kwargs)
            return value * use_unit.ureg(unit)
        return wrapper_user_unit
    return decorator_use_unit

@use_unit("meters per second")
def average_speed(distance, duration):
    return distance / duration

# pass arguments distance = 100 and duration = 9.58 to average_speed function
# decorated by use_unit decorator function, in order to get average speed
# in meters per second. Assign this value to variable bolt
bolt = average_speed(100, 9.58)

# print out value of bolt
print(bolt)

# convert bolt to km/hr
print(bolt.to("km per hour"))

# convert bolt to mph and list only magnitude
print(bolt.to("mph").m)