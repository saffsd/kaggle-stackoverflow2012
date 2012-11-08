"""
data2vw.py

Marco Lui, October 2012
based on csv2vw.py and extract.py
"""

import argparse
import csv
import nltk
from collections import Counter, Mapping, Sequence, defaultdict
from itertools import groupby
import re
import numpy as np
from multiprocessing import Pool
from dateutil import parser as dateparser

import os

CLASSES = [ 'not a real question', 'not constructive', 'off topic', 'open', 'too localized']
status = dict( (k, str(i+1)) for i,k in enumerate(CLASSES))

RE_NONALNUM = re.compile(r'\W+')
RE_NONANS   = re.compile(r'[^\w\s]+')
RE_DIGIT    = re.compile(r'\d+')
RE_URL      = re.compile(r'https?://')
RE_NONWORD  = re.compile(r'[A-Z\d]+')

def norm(string):
  return RE_NONANS.sub('', string).lower()

def norm_tag(string):
  return RE_NONALNUM.sub('', string).lower()

def norm_bow_tokens(bow):
  # Remove nonalphanumeric components of the word, due to VW format restrictions
  retval = defaultdict(int)
  for key, value in bow.iteritems():
    key = RE_NONALNUM.sub('',key)
    if key:
      retval[key] += value
  return retval

def ngram(seq, n):
  dist = Counter( '_'.join(seq[pos:pos+n]) for pos in xrange(len(seq) - n + 1) )
  return dist


def data2vw(data, name, weight=None):
  """
  Turn a mapping to a vw-format string
  """
  # ensure only alphanumeric lowercase in name
  name = norm_tag(name)

  if isinstance(data, Mapping):
    features = ' '.join('{0}:{1}'.format(*i) for i in data.iteritems())
  elif isinstance(data, Sequence):
    features = ' '.join(data)
  else:
    raise ValueError("don't know how to handle data of type {0}".format(type(data)))

  ident = name if weight is None else "{0}:{1}".format(name,weight)
  retval = "|{0} {1}".format(ident, features)
  return retval

def ratio(x,y):
  if y != 0:
    return x / float(y)
  else:
    return 0

def row2wv(row):
  """
  Conver a single row to vw format
  """
  post_id = row['PostId']
  try:
    post_status = status[row['OpenStatus']]
  except KeyError:
    # no OpenStatus, must be a test file
    post_status = '0'

  title = row['Title']
  body = row['BodyMarkdown']
  tags = [norm_tag(row["Tag%d"%i]) for i in range(1,6) if row["Tag%d"%i]]

  lines = body.splitlines()
  code = []
  text = []
  sents = []
  # Divide post into code and text blocks
  for is_code, group in groupby(lines, lambda l: l.startswith('    ')):
    (code if is_code else text).append('\n'.join(group))
    
  # Let's build some features!
  user = {}
  stats = defaultdict(dict)
  stats['num']['sent'] = 0
  stats['num']['question'] = 0
  stats['num']['exclam'] = 0
  stats['num']['period'] = 0
  stats['num']['initcap'] = 0
  stats['num']['istart'] = 0
  stats['num']['url'] = 0
  stats['num']['digit'] = 0
  stats['num']['nonword'] = 0
  body_words = set()

  firstsentwords = None
  lastsentwords = None
  for t in text:
    for sent in nltk.sent_tokenize(t):
      stats['num']['sent'] += 1
      ss = sent.strip()
      if ss:
        if ss.endswith('?'):
          stats['num']['question'] += 1
        if ss.endswith('!'):
          stats['num']['exclam'] += 1
        if ss.endswith('.'):
          stats['num']['period'] += 1
        if ss.startswith('I '):
          stats['num']['istart'] += 1
        if ss[0].isupper():
          stats['num']['initcap'] += 1

      words = nltk.word_tokenize(norm(sent))

      # We track the set of words of the first and last sentences
      lastsentwords = set(words)
      if firstsentwords is None:
        firstsentwords = lastsentwords
      body_words |= lastsentwords
      sents.append(ss)

    stats['num']['digit'] += len(RE_DIGIT.findall(t))
    stats['num']['url'] += len(RE_URL.findall(t))
    stats['num']['nonword'] += len(RE_NONWORD.findall(t))
      
  stats['num']['finalthanks'] = 1 if text and 'thank' in text[-1].lower() else 0

  # NOTE: stopping didn't work. 
  body_words = list(body_words)
  firstsentwords = list(firstsentwords) if firstsentwords else []
  lastsentwords = list(lastsentwords) if lastsentwords else []

  # TODO: See if we can get some use of out the code blocks
  """
  code_ngrams = Counter()
  for c in code:
    ngram = nltk.word_tokenize(norm(c))
    #ngram = base64ngram(c, 4)
    code_ngrams.update(ngram)
  """
  title_words = nltk.word_tokenize(norm(title))
  title_words = list(set(title_words))

  post_t = dateparser.parse(row['PostCreationDate'])
  user_t = dateparser.parse(row['OwnerCreationDate'])

  # Some stats about the user
  user['age'] = (post_t - user_t).total_seconds()
  user['reputation'] = int(row['ReputationAtPostCreation'])
  user['good_posts'] = int(row['OwnerUndeletedAnswerCountAtPostTime'])
  user['userid']     = row['OwnerUserId']

  stats['num']['codeblock'] = len(code)
  stats['num']['textblock'] = len(text)
  stats['num']['lines']     = len(lines)
  stats['num']['tags']      = len(tags)
  stats['len']['title']     = len(title)
  stats['len']['text']      = sum(len(t) for t in text)
  stats['len']['code']      = sum(len(c) for c in code)
  stats['len']['firsttext'] = len(text[0]) if text else 0
  stats['len']['firstcode'] = len(code[0]) if code else 0
  stats['len']['lasttext']  = len(text[-1]) if text else 0
  stats['len']['lastcode']  = len(code[-1]) if code else 0
  stats['ratio']['tc']      = ratio(stats['len']['text'],stats['len']['code'])
  stats['ratio']['ftc']     = ratio(stats['len']['firsttext'],stats['len']['firstcode'])
  stats['ratio']['ftext']   = ratio(stats['len']['firsttext'],stats['len']['text'])
  stats['ratio']['fcode']   = ratio(stats['len']['firstcode'],stats['len']['code'])
  stats['ratio']['qsent']   = ratio(stats['num']['question'],stats['num']['sent'])
  stats['ratio']['esent']   = ratio(stats['num']['exclam'],stats['num']['sent'])
  stats['ratio']['psent']   = ratio(stats['num']['period'],stats['num']['sent'])
  stats['mean']['code']     = np.mean([len(c) for c in code]) if code else 0
  stats['mean']['text']     = np.mean([len(t) for t in text]) if text else 0
  stats['mean']['sent']     = np.mean([len(s) for s in sents]) if sents else 0

  # construct the vw-format line
  header = "{0} 1.0 {1}".format(post_status, post_id)
  segments = [
    ##data2vw(code_ngrams.keys(),'code'),
    data2vw(stats['num'],'xsnum'),
    data2vw(stats['len'],'yslen'),
    data2vw(stats['ratio'],'zsratio'),
    data2vw(stats['mean'],'wsmean'),
    data2vw(user,'user'),
    data2vw(title_words,'titlewords'),
    data2vw(body_words,'bodywords'),
    data2vw(tags,'vtags'),
    #data2vw(firstsentwords,'firstsent'),
    #data2vw(lastsentwords,'lastsent'),
  ]
  outline = header + ' '.join(segments) + '\n'

  """
  if post_status != '4':
    print "===== {0} : {1} =====".format(post_id, CLASSES[int(post_status) - 1])
    for s in sents:
      if s:
        print s
    import ipdb; ipdb.set_trace()
  """

  return outline

from itertools import imap

from timeit import default_timer
class Timer(object):
  def __init__(self):
    self.timer = default_timer
    self.start = None
    self.end = None

  def __enter__(self):
    self.start = self.timer()
    self.end = None
    return self

  def __exit__(self, *args):
    self.end = self.timer()

  @property
  def elapsed(self):
    now = self.timer()
    if self.end is not None:
      self.end - self.start
    else:
      return now - self.start

  def rate(self, count):
    now = self.timer()
    if self.start is None:
      raise ValueError("Not yet started")

    return count / (now - self.start)

if __name__ == "__main__":
  parser = argparse.ArgumentParser("convert from kaggle format to vw format")
  parser.add_argument("input")
  parser.add_argument("output")
  args = parser.parse_args()

  reader = csv.DictReader( open(args.input) )
  pool = Pool()

  with open(args.output, 'w') as outf:
    with Timer() as t:
      for i,outline in enumerate(pool.imap(row2wv, reader, chunksize=100)):
      #for i,outline in enumerate(imap(row2wv, reader)):
        outf.write( outline )

        if i % 10000 == 0:
          print "{0} lines in {1}s ({2} lines/s)".format(i, t.elapsed, t.rate(i))

