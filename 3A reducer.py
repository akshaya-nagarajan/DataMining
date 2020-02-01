#!/usr/bin/python
from operator import itemgetter
import sys

current_count_5000 = 1
current_count_10000 = 0
word = None
l = 0
# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()

    # parse the input we got from mapper.py
    word = line.split('\t')
    #print(type(word[0]))
    # this IF-switch only works because Hadoop sorts map output
    # by key (here: word) before it is passed to the reducer
    #if current_word == word:
    l += len(line)
    
    if word[0].strip() and l < 10353:
       if float(word[0]) < 5000.0:   
           current_count_5000 += 1
       elif float(word[0]) >= 10000.0 and float(word[0]) <= 20000.0:
           current_count_10000 += 1
#print(current_count_5000, current_count_10000)
print '%s\t%s' % ('Number of rows in 9th_12th grade less than 5000',current_count_5000)
print '%s\t%s' % ('Number of rows in 9th_12th grade between 10000 and 20000', current_count_10000)
