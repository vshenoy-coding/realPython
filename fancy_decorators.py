#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 05:07:19 2022

@author: vms
"""

class Circle:
    def __init__(self, radius):
        self._radius = radius
        #self._height = height


    @property
    def radius(self):
        """Get value of radius"""
        return self._radius
    
    # @property
    # def height(self):
    #     """Get value of height"""
    #     return self._height

    @radius.setter
    def radius(self, value):
        """Set radius, raise error if negative"""
        if value >= 0:
            self._radius = value
        else:
            raise ValueError("Radius must be positive")
    
    
    # @height.setter
    # def height(self, value):
    #     if value >= 0:
    #         self._height = value
    #     else:
    #         raise ValueError("Height must be positive")
    
    @property
    def area(self):
        """Calculate area inside circle"""
        return self.pi() * self.radius**2

    def cylinder_volume(self, height):
        
        if height <= 0:
            raise ValueError("Height must be positive")
          
        """Calculate volume of cylinder with circle as base"""
        return self.area * height


    @classmethod
    def unit_circle(cls):
        """Factory method creating a circle with radius 1"""
        return cls(1)

    @staticmethod
    def pi():
        """Value of π, could use math.pi instead though"""
        return 3.1415926535

# In this class:

# .cylinder_volume() is a regular method.

# .radius is a mutable property: it can be set to a different value. 
# However, by defining a setter method, we can do some error testing to make 
# sure it’s not set to a nonsensical negative number. 
# Properties are accessed as attributes without parentheses.

# .area is an immutable property: 
# properties without .setter() methods can’t be changed. 
# Even though it is defined as a method, 
# it can be retrieved as an attribute without parentheses.

# .unit_circle() is a class method. It’s not bound to one particular instance 
# of Circle. Class methods are often used as factory methods that can create 
# specific instances of the class.

# .pi() is a static method. It’s not really dependent on the Circle class, 
# except that it is part of its namespace. 
# Static methods can be called on either as an instance or the class.


##Added error checking code for height

c = Circle(5)
#print(c.radius)
#print(c.area)

#c.radius = 2
#print(c.area)



#c.area = 100 
# error, .area is an immutable property of class Circle, so its attribute 
# can't be set

# height must be positive value, otherwise ValueError is thrown
#print(c.cylinder_volume(height=-4))

# radius can't be set to negative value
#c.radius = -1

#print(Circle.pi())

###
# Uncomment decorated function lines and print statements to see results.
# Make sure to avoid commenting out muliple decorator functions with the same
# name, as that could cause conflicts and cause the wrong line of code to run.
import functools

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


# def timer(func):
#     """Print the runtime of the decorated function"""
#     @functools.wraps(func)
#     def wrapper_timer(*args, **kwargs):
#         start_time = time.perf_counter()    # 1
#         value = func(*args, **kwargs)
#         end_time = time.perf_counter()      # 2
#         run_time = end_time - start_time    # 3
#         print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
#         return value
#     return wrapper_timer

# @timer
# def waste_some_time(num_times):
#     for _ in range(num_times):
#         sum([i**2 for i in range(10000)])

###
# from decorators import debug, timer

# class TimeWaster:
#     @debug
#     def __init__(self, max_num):
#         self.max_num = max_num

#     @timer
#     def waste_time(self, num_times):
#         for _ in range(num_times):
#             sum([i**2 for i in range(self.max_num)])

###
# class decorators
# @time only measures time it takes to instantiate the class, TimeWaster
# from decorators import timer

# @timer
# class TimeWaster:
#     def __init__(self, max_num):
#         self.max_num = max_num

#     def waste_time(self, num_times):
#         for _ in range(num_times):
#             sum([i**2 for i in range(self.max_num)])
            
            
###
# import functools
def do_twice(func):
    @functools.wraps(func)
    def wrapper_do_twice(*args, **kwargs):
        func(*args, **kwargs)
        func(*args, **kwargs)
    return wrapper_do_twice

#from decorators import debug, do_twice
# ^ commented out this line, as it resulted in greet function being
# named as 'wrapper_do_twice' instead of 'greet' when checking with 
# .__greet__ in console

@debug
@do_twice
def greet(name=None):
    name = " " if name is None else name
    print(f"Hello, {name}")
    
# greet("Eva")

###
# decorators with arguments


def repeat(num_times):
    def decorator_repeat(func):
        @functools.wraps(func)
        def wrapper_repeat(*args, **kwargs):
            for _ in range(num_times):
                value = func(*args, **kwargs)
            return value
        return wrapper_repeat
    return decorator_repeat

@repeat(num_times=4)
def greet(name=None):
    name = " " if name is None else name
    print(f"Hello {name}")

#greet("World")


# when a decorator uses arguments, you need an extra outer function
# add the _func parameter

def repeat(_func=None, *, num_times=2):
    def decorator_repeat(func):
        @functools.wraps(func)
        def wrapper_repeat(*args, **kwargs):
            for _ in range(num_times):
                value = func(*args, **kwargs)
            return value
        return wrapper_repeat

    if _func is None:
        return decorator_repeat        #3
    else:
        return decorator_repeat(_func) #4
    
    
# If decorator, repeat has been called without arguments, 
# the decorated function will be passed in as _func. 

# If the decorator has been called with arguments, then _func will be None, 
# and some of the keyword arguments may have been changed from their default values. 
# The * in the argument list means that the remaining arguments can’t be 
# called as positional arguments.

# 3 In this case, the decorator was called with arguments (i.e.: num_times). 
# Return a decorator function that can read and return a function.

# 4 In this case, the decorator was called without arguments. 
# Apply the decorator to the function immediately.

@repeat
def say_greet():
    print("Greetings")


@repeat(num_times=3)
def greet(name=None):
    name = " " if name is None else name
    print(f"Hello {name}")

#print(say_greet())

#print(greet("Paul"))

###
# Stateful Decorators
def count_calls(func):
    @functools.wraps(func)
    def wrapper_count_calls(*args, **kwargs):
        wrapper_count_calls.num_calls += 1
        print(f"Call {wrapper_count_calls.num_calls} of {func.__name__!r}")
        return func(*args, **kwargs)
    wrapper_count_calls.num_calls = 0
    return wrapper_count_calls

@count_calls
def say_whee():
    print("Whee!")

#print(say_whee())
#print(say_whee())
#print(say_whee.num_calls)

###
# Classes as Decorators
class Counter:
    def __init__(self, start=0):
        self.count = start

    # for a class instance to be callable, you implement the .__call__ method
    def __call__(self):
        self.count += 1
        print(f"Current count is {self.count}")

# The .__call__() method is executed each time you try to call 
# an instance of the class

counter = Counter()
counter()
counter()
#print(counter.count)



class CountCalls:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.num_calls = 0

    def __call__(self, *args, **kwargs):
        self.num_calls += 1
        print(f"Call {self.num_calls} of {self.func.__name__!r}")
        return self.func(*args, **kwargs)

@CountCalls
def say_whee():
    print("Whee!")
    
#print(say_whee())
#print(say_whee())
#print(say_whee.num_calls)