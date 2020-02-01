#!/usr/bin/python
from operator import itemgetter
import sys

current_word = None
current_count = 0
word = None
finalWord = ''
lowerCaseList = []
wordArrayWithoutDuplicates = []
# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace - Data preprocessing
    line = line.strip()

    # parse the input we got from mapper.py
    word, count = line.split('\t', 1)

    # convert count (currently a string) to int
    try:
        count = int(count)
    except ValueError:
        # count was not a number, so silently
        # ignore/discard this line
        continue

    # this IF-switch only works because Hadoop sorts map output
    # by key (here: word) before it is passed to the reducer
    if current_word == word:
        current_count += count
    else:
        if current_word:
            # write result to STDOUT
            print '%s\t%s' % (current_word, current_count)
        current_count = count
        current_word = word
        wordArrayWithoutDuplicates.append(word)
    if not word:
      print '%s' % ("Not a string")
    
    if word.isalnum():
      finalWord = finalWord + ''.join(e for e in word)

    lowerCaseList.append(word.lower())
# Removing duplicates
print '%s\t%s' % ("Removing duplicates", wordArrayWithoutDuplicates)
# Removing special characters
print '%s\t%s' % ("Removing special characters",finalWord)
# converting to lowercase
print '%s\t%s' % ("converting to lowercase",lowerCaseList)
