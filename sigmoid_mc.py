'read vw raw predictions file, compute and normalize probabilities, write in submission format'

import sys, csv, math
import argparse
import numpy as np

def sigmoid(x):
  try:
    return 1 / (1 + math.exp(-x))
  except Exception:
    import ipdb;ipdb.set_trace()
  
  
def normalize( predictions ):
  s = sum( predictions )
  normalized = []
  for p in predictions:
    normalized.append( p / s )
  return normalized  
  
###  
if __name__ == "__main__": 
  parser = argparse.ArgumentParser()
  parser.add_argument('input')
  parser.add_argument('output')
  parser.add_argument('-i','--id',action='store_true', default=False, help='include post id')
  args = parser.parse_args()

  i = open( args.input )
  o = open( args.output, 'wb' )

  reader = csv.reader( i, delimiter = " " )
  writer = csv.writer( o )

  for line in reader:
    
    post_id = reader.next()[1]
    
    probs = []
    for element in line:
      prediction = element.split( ":" )[1]
      prob = sigmoid( float( prediction ))
      probs.append( prob )
    
    pd = normalize( probs )
    
    if args.id:
      writer.writerow( [post_id] + list(pd) )
    else:
      writer.writerow( list(pd) )
