"""
Tabulate user-level statistics
We want to know how many questions of
each type each user has.

Marco Lui, October 2012
"""

import argparse
import csv
import operator

from collections import Counter, defaultdict

from data2vw import norm, Timer, status


if __name__ == "__main__":
  parser = argparse.ArgumentParser("user stats of SO data")
  parser.add_argument('input')
  parser.add_argument('output')

  args = parser.parse_args()

  reader = csv.DictReader( open(args.input) )

  user_posts = defaultdict(lambda: defaultdict(int))

  with Timer() as t:
    for i,row in enumerate(reader):
      uid = row['OwnerUserId']
      post_status = status[row['OpenStatus']]
      user_posts[uid][post_status] += 1

      if i and i % 10000 == 0: 
        print '{0} lines in {1}s ({2} l/s)'.format(i, t.elapsed, t.rate(i))
    print '{0} lines in {1}s ({2} l/s)'.format(i+1, t.elapsed, t.rate(i+1))


  #output stage
  with open(args.output, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(('OwnerUserId',) + tuple('Status{0}'.format(k) for k in range(6)))
    writer.writerows((u,) + tuple(v[str(k)] for k in range(6)) for u,v in user_posts.iteritems())

 

