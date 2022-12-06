import sys

d = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth'}

# create empty dictionary e for populating
e = {}
# create keys in range of 1 to 9 inclusive
keys = [i for i in range(1,10)]
# create corresponding values
values = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh",
          "eigth", "ninth"]
print(values)
print(keys)

e = dict(list(enumerate(values, start=1)))
print(e)
#print(type(e)) # e is of class 'dict'

# alternate way to create dictionaries, with dict() and zip() methods
#e1 = dict(zip(keys,values))

# print populated dictionary e with key and value pairs
#print(e1)
#print(type(e1)) # e1 is of class 'dict'


# code to print out statements in multi_yield function is in a separate, later
# line
def multi_yield(*args, **kwargs):
    for i in d.values():
        yield_str = "This will print the " + i +  " string"
        yield yield_str
    
multi_gen = multi_yield()
#print(type(multi_yield)) # multi_yield is of class 'function'
#print(type(multi_gen)) # multi_gun is of class 'generator'

# code to print out statements is in multi_func function definition itself
def multi_func(*args, **kwargs):
    for i in d.values():
        func_str = "This will print the " + i + " string"
        print(func_str)
        
multi_obj = multi_func()
#print(type(multi_func)) # multi_func is of class 'function'
#print(type(multi_obj)) # multi_obj is of class 'NoneType'

# check whether multi_gen has smaller size than multi_obj
#print(sys.getsizeof(multi_gen))
#print(sys.getsizeof(multi_obj))

x = 0
if (sys.getsizeof(multi_obj) > sys.getsizeof(multi_gen)):
    x = x + 1 # x will get incremented if multi_gen uses less memory than multi_obj
else:
    print("Generator function multi_gen does not use less memory.")
#print(x)

# print out statements in multi_gen function
for i in d.values():
    print(next(multi_gen))
    
    
letters = ["a", "b", "c", "y"]
it = iter(letters)


# print out each iterable in variable it 
# prints out a b c y 
while True:
    try:
        letter = next(it)
    except StopIteration:
        break
    print(letter)

letters = ["a", "b", "c", "y"]
it = iter(letters)
# print out list of iterables
#print(list(it)) # prints out a b c y

# enumerate variable it, use list method on the enuemrate object, and then assign result to variable a
a = list(enumerate(it,start=1))
#print(a) 
#print(type(a)) # a is of class 'list'

# if list method was not used, class with be of type enumerate
a2 = enumerate(it, start=1)
#print(a2)
#print(type(a2)) # a2 is of class 'enumerate'


# alternate way with zip method
a1 = list(zip([i for i in range(1,5)], letters))
#print(a1)
#print(type(a1)) # a1 is of class 'list'

# enumerate object with indices starting at 3 and which are multiples of 3
index = []
for count, item in enumerate(a, start=1):
    count = count *3
    index.append(count)

#print(index)
# create enumerate object that starts at 3 and increases by multiples of 3,
# through using the list method on the zip object created with arguments index and letters
# for count and item
b = list(zip(index, letters))
#print(b)
#print(type(b)) # b is of class 'list'


# shorter form of creating enumerate object unlike for loop above.
# index starts at 3 and increases by multiples of 3
c = list(zip([3,6,9,12],letters))
#print(c)
#print(type(c)) # c is of class 'list'

# or alternatively 
c1 = list(zip([i*3 for i in range(1,5)],letters))
#print(c1)
#print(type(c1)) # c1 is of class 'list'

# enumerate object where indices start at 3 and increase by 1
g = list(enumerate(letters, start = 3))
#print(g)
#print(type(g)) # g is of class 'list'

# or alternatively
g1 = list(zip([i for i in range(3,7)], letters))
#print(g1)
#print(type(g1)) # g1 is of class 'list'

# a, a1, b, c, c1, g, and g1 all have different ids
# print(id(a))
# print(id(a1))
# print(id(b))
# print(id(c))
# print(id(c1))
# print(id(g))
# print(id(g1))

if id(a) is not id(a1) is not id(b) is not id(c) is not id(c1) is not id(g) is not id(g1):
    print(True)
    