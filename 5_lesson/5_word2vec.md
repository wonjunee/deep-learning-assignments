
Deep Learning
=============

Assignment 5
------------

The goal of this assignment is to train a Word2Vec skip-gram model over [Text8](http://mattmahoney.net/dc/textdata) data.


```python
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
```

Download the data from the source website if necessary.


```python
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)
```

    Found and verified text8.zip


Read the data into a string.


```python
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data
  
words = read_data(filename)
print('Data size %d' % len(words))
```

    Data size 17005207


Build the dictionary and replace rare words with UNK token.


```python
vocabulary_size = 50000

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.
```

    Most common words (+UNK) [['UNK', 418391], ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764)]
    Sample data [5239, 3084, 12, 6, 195, 2, 3137, 46, 59, 156]


Function to generate a training batch for the skip-gram model.


```python
data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print(np.shape(batch))
    print(np.shape(labels))
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
```

    data: ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first']
    (8,)
    (8, 1)
    
    with num_skips = 2 and skip_window = 1:
        batch: ['originated', 'originated', 'as', 'as', 'a', 'a', 'term', 'term']
        labels: ['as', 'anarchism', 'originated', 'a', 'term', 'as', 'a', 'of']
    (8,)
    (8, 1)
    
    with num_skips = 4 and skip_window = 2:
        batch: ['as', 'as', 'as', 'as', 'a', 'a', 'a', 'a']
        labels: ['originated', 'term', 'anarchism', 'a', 'as', 'term', 'of', 'originated']


Train a skip-gram model.


```python
batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. 
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):

  # Input data.
  train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  
  # Variables.
  embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  softmax_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                         stddev=1.0 / math.sqrt(embedding_size)))
  softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Model.
  # Look up embeddings for inputs.
  embed = tf.nn.embedding_lookup(embeddings, train_dataset)
  # Compute the softmax loss, using a sample of the negative labels each time.
  loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                               train_labels, num_sampled, vocabulary_size))

  # Optimizer.
  optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
  
  # Compute the similarity between minibatch examples and all embeddings.
  # We use the cosine distance:
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
```


```python
num_steps = 100001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  average_loss = 0
  for step in range(num_steps):
    batch_data, batch_labels = generate_batch(
      batch_size, num_skips, skip_window)
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += l
    if step % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step %d: %f' % (step, average_loss))
      average_loss = 0
    # note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log = '%s %s,' % (log, close_word)
        print(log)
  final_embeddings = normalized_embeddings.eval()
```

    Initialized
    Average loss at step 0: 8.193994
    Nearest to first: medians, factoring, merrill, dmc, popularizer, ahijah, comore, inxs,
    Nearest to when: duplicating, subfields, swordfish, agilent, eur, cognomen, appreciable, citro,
    Nearest to is: subservient, holiness, magnate, emptiness, disintegrating, rory, landsat, theorem,
    Nearest to their: online, eris, jarman, tertullian, socrates, stench, casing, ascetics,
    Nearest to zero: devo, normale, deformation, virility, negativity, dunn, cataract, verdi,
    Nearest to new: cements, countering, slovakia, asia, cline, trier, kilby, grandeur,
    Nearest to to: dye, staining, flights, hacked, devoid, dynamics, fetishism, defeats,
    Nearest to or: serene, usurped, injera, reset, cacao, howlin, avatar, herodotus,
    Nearest to the: pits, halfback, falsity, synthesize, gringo, prix, dg, scientology,
    Nearest to during: butcher, tennessee, nomadic, notational, chiefly, colonized, ripken, subdivision,
    Nearest to six: held, perceptive, shellfish, lindo, overestimated, bribes, ia, destry,
    Nearest to state: diarrhoea, sen, abridgment, anoint, sketch, sumatra, iridescent, biomes,
    Nearest to in: raped, differs, annul, install, edt, stocks, deicide, fula,
    Nearest to over: unconsciousness, yitzchak, milligan, greenman, para, benson, wields, codecs,
    Nearest to i: fever, thermodynamically, leaks, uighur, elijah, symonds, golems, discursive,
    Nearest to known: applying, superconductor, phillips, situ, numismatic, basilides, causative, plantagenet,
    Average loss at step 2000: 4.362755
    Average loss at step 4000: 3.863167
    Average loss at step 6000: 3.792338
    Average loss at step 8000: 3.683231
    Average loss at step 10000: 3.617616
    Nearest to first: second, popularizer, next, calorie, bahasa, freeciv, powder, aleinu,
    Nearest to when: where, agilent, appreciable, duplicating, subfields, cerf, eur, swordfish,
    Nearest to is: was, are, has, be, kidding, attained, tutor, protective,
    Nearest to their: his, its, the, agreeable, s, online, this, socrates,
    Nearest to zero: nine, eight, five, six, seven, three, four, two,
    Nearest to new: slovakia, kilby, kaplan, methuen, tic, asia, trier, ucs,
    Nearest to to: would, samos, circumscription, flights, psychiatry, not, paige, cottingley,
    Nearest to or: and, than, cockpit, lapsed, pressurized, keen, square, spaniard,
    Nearest to the: a, this, his, its, their, zz, synthesize, an,
    Nearest to during: subdivision, through, nomadic, coming, booklet, in, following, metroid,
    Nearest to six: eight, seven, five, four, three, nine, zero, two,
    Nearest to state: espn, anoint, biomes, malaya, childe, sen, loaders, bundle,
    Nearest to in: on, of, at, from, through, with, dhimmi, for,
    Nearest to over: milligan, apo, hamilcar, latex, diem, rationalist, berger, klaproth,
    Nearest to i: symonds, elijah, fever, azathoth, cypher, sends, fim, flatly,
    Nearest to known: basilides, entry, an, esperantists, nashe, used, situ, sess,
    Average loss at step 12000: 3.601654
    Average loss at step 14000: 3.575543
    Average loss at step 16000: 3.408424
    Average loss at step 18000: 3.460736
    Average loss at step 20000: 3.537892
    Nearest to first: next, second, truman, brooker, bahasa, last, freeciv, popularizer,
    Nearest to when: where, if, hydrodynamics, during, was, agilent, recognising, through,
    Nearest to is: was, are, has, but, does, were, stosunku, be,
    Nearest to their: his, its, the, some, her, any, these, socrates,
    Nearest to zero: five, six, seven, four, three, nine, eight, two,
    Nearest to new: slovakia, asia, trier, kaplan, kilby, tic, cline, obedient,
    Nearest to to: towards, propagated, hacked, would, obsessed, not, can, will,
    Nearest to or: and, than, for, hydrothermal, benitez, keen, chisinau, equus,
    Nearest to the: its, their, his, an, territorial, this, realpolitik, any,
    Nearest to during: at, following, subdivision, metroid, in, through, when, nomadic,
    Nearest to six: eight, five, seven, four, nine, three, two, zero,
    Nearest to state: loaders, malaya, childe, espn, biomes, largest, lysosomes, anoint,
    Nearest to in: at, with, on, through, from, dhimmi, and, for,
    Nearest to over: milligan, logarithmic, rationalist, poseidon, apo, hong, diem, aleutians,
    Nearest to i: ii, symonds, cii, elijah, kelly, azathoth, rip, miquelon,
    Nearest to known: used, defined, seen, well, leftists, nashe, annoyed, avestan,
    Average loss at step 22000: 3.497390
    Average loss at step 24000: 3.489797
    Average loss at step 26000: 3.480174
    Average loss at step 28000: 3.484245
    Average loss at step 30000: 3.499509
    Nearest to first: next, second, last, truman, halting, beverage, initialism, brooker,
    Nearest to when: where, if, during, though, while, was, before, hydrodynamics,
    Nearest to is: was, has, are, does, be, became, attained, if,
    Nearest to their: its, his, the, her, these, some, any, our,
    Nearest to zero: five, eight, seven, four, six, three, nine, two,
    Nearest to new: slovakia, kaplan, tic, graphics, argumentation, asia, vieux, total,
    Nearest to to: would, will, can, propagated, could, must, svg, agricultural,
    Nearest to or: and, suzanne, than, ducati, prosper, but, viollet, glottal,
    Nearest to the: their, its, his, some, a, this, angora, her,
    Nearest to during: at, when, after, in, through, following, metroid, diver,
    Nearest to six: eight, four, seven, nine, five, three, two, zero,
    Nearest to state: malaya, loaders, lysosomes, biomes, childe, marked, largest, audible,
    Nearest to in: at, during, dhimmi, on, from, under, and, of,
    Nearest to over: milligan, logarithmic, rationalist, poseidon, latex, into, aleutians, macross,
    Nearest to i: ii, we, g, azathoth, carr, spooky, kelly, symonds,
    Nearest to known: used, well, defined, such, seen, leftists, annoyed, nashe,
    Average loss at step 32000: 3.501771
    Average loss at step 34000: 3.494286
    Average loss at step 36000: 3.455392
    Average loss at step 38000: 3.305419
    Average loss at step 40000: 3.429850
    Nearest to first: second, next, last, truman, bahasa, conceits, popularizer, lecoq,
    Nearest to when: if, before, while, during, where, though, after, hydrodynamics,
    Nearest to is: was, does, are, has, be, if, propelled, maroger,
    Nearest to their: its, his, her, the, any, our, these, other,
    Nearest to zero: seven, five, eight, three, nine, six, two, four,
    Nearest to new: slovakia, renunciation, bionic, statement, trier, icebergs, monkeys, alien,
    Nearest to to: must, will, would, towards, might, connectedness, propagated, could,
    Nearest to or: and, than, benitez, a, luz, budge, vg, astm,
    Nearest to the: their, its, any, each, a, this, some, his,
    Nearest to during: when, through, after, at, following, around, on, diver,
    Nearest to six: five, four, three, seven, eight, two, nine, one,
    Nearest to state: loaders, malaya, stylish, botham, bundle, lysosomes, republic, rooks,
    Nearest to in: of, viewership, dhimmi, melodic, and, from, for, within,
    Nearest to over: latex, aleutians, poseidon, across, into, altenberg, marjoram, alleviating,
    Nearest to i: we, ii, they, you, t, spooky, he, g,
    Nearest to known: used, defined, such, well, sucre, khazaria, seen, monsignor,
    Average loss at step 42000: 3.435585
    Average loss at step 44000: 3.451757
    Average loss at step 46000: 3.452523
    Average loss at step 48000: 3.354169
    Average loss at step 50000: 3.381607
    Nearest to first: second, next, last, third, freeciv, truman, same, latter,
    Nearest to when: if, where, while, after, before, during, though, although,
    Nearest to is: was, has, are, does, be, became, amortized, repetitive,
    Nearest to their: its, his, her, the, our, any, characterizes, these,
    Nearest to zero: five, eight, seven, six, four, three, nine, two,
    Nearest to new: slovakia, statement, renunciation, wide, conceiving, prussian, labonte, cory,
    Nearest to to: would, connectedness, algal, rtf, cloning, towards, cottingley, can,
    Nearest to or: and, than, glottal, suzanne, benitez, lifestyles, like, jacinto,
    Nearest to the: its, their, his, this, some, a, any, angora,
    Nearest to during: after, in, when, at, through, following, under, against,
    Nearest to six: seven, eight, four, nine, three, five, one, two,
    Nearest to state: stylish, loaders, malaya, states, lysosomes, bundle, largest, botham,
    Nearest to in: during, of, dhimmi, from, since, within, on, therapists,
    Nearest to over: aleutians, poseidon, diem, latex, about, measuring, batting, carlisle,
    Nearest to i: we, you, ii, they, azathoth, cerevisiae, t, rip,
    Nearest to known: used, defined, such, well, regarded, classified, seen, sucre,
    Average loss at step 52000: 3.435977
    Average loss at step 54000: 3.427845
    Average loss at step 56000: 3.437118
    Average loss at step 58000: 3.396612
    Average loss at step 60000: 3.393462
    Nearest to first: next, second, last, third, truman, only, best, sailor,
    Nearest to when: if, after, before, where, although, while, during, though,
    Nearest to is: was, has, are, does, became, tutor, be, contains,
    Nearest to their: its, his, her, the, our, these, some, your,
    Nearest to zero: five, four, six, seven, eight, nine, three, two,
    Nearest to new: slovakia, conceiving, prussian, pacheco, gundobad, focussing, channeled, wide,
    Nearest to to: towards, would, cottingley, will, legged, not, rtf, propagated,
    Nearest to or: and, than, glottal, non, benitez, vg, like, bmx,
    Nearest to the: its, their, a, his, this, desai, any, consecutive,
    Nearest to during: after, in, when, following, before, although, at, since,
    Nearest to six: four, eight, five, nine, seven, three, zero, one,
    Nearest to state: malaya, loaders, states, lysosomes, property, government, city, republic,
    Nearest to in: during, within, since, at, of, dhimmi, from, into,
    Nearest to over: around, poseidon, into, aleutians, about, diem, measuring, across,
    Nearest to i: we, you, ii, they, t, rhombic, rip, flintlock,
    Nearest to known: used, defined, such, well, regarded, classified, khazaria, possible,
    Average loss at step 62000: 3.237064
    Average loss at step 64000: 3.258351
    Average loss at step 66000: 3.401028
    Average loss at step 68000: 3.395535
    Average loss at step 70000: 3.360291
    Nearest to first: second, next, last, third, same, truman, sailor, popularizer,
    Nearest to when: if, where, while, before, after, though, because, however,
    Nearest to is: was, has, be, are, makes, became, synclavier, requires,
    Nearest to their: its, his, her, the, our, your, these, personal,
    Nearest to zero: five, six, four, eight, seven, three, nine, two,
    Nearest to new: slovakia, pacheco, renunciation, tic, channeled, conceiving, gundobad, trump,
    Nearest to to: will, must, towards, in, can, could, would, connectedness,
    Nearest to or: and, while, glottal, than, any, benitez, vg, but,
    Nearest to the: their, a, any, its, this, each, these, or,
    Nearest to during: after, in, before, until, at, following, through, under,
    Nearest to six: seven, eight, five, four, nine, three, zero, two,
    Nearest to state: states, lysosomes, city, provincial, loaders, malaya, feasibility, government,
    Nearest to in: within, during, on, dhimmi, between, carthusian, to, keeps,
    Nearest to over: around, poseidon, aleutians, diem, explodes, routine, defining, logarithmic,
    Nearest to i: we, you, ii, g, rip, puzzled, they, kenobi,
    Nearest to known: used, defined, regarded, classified, seen, such, well, considered,
    Average loss at step 72000: 3.376012
    Average loss at step 74000: 3.349915
    Average loss at step 76000: 3.316679
    Average loss at step 78000: 3.351037
    Average loss at step 80000: 3.376237
    Nearest to first: second, next, last, third, popularizer, same, sailor, best,
    Nearest to when: before, if, after, while, though, where, during, although,
    Nearest to is: was, has, are, be, dissociation, became, requires, although,
    Nearest to their: its, his, her, our, your, the, ravi, my,
    Nearest to zero: five, six, seven, eight, four, three, nine, two,
    Nearest to new: slovakia, pacheco, conceiving, prussian, argumentation, imaging, craftsman, distaste,
    Nearest to to: will, towards, must, would, rtf, should, could, inductee,
    Nearest to or: and, jacinto, than, while, cfp, glottal, benitez, vg,
    Nearest to the: its, his, this, their, cebit, a, skyscrapers, riefenstahl,
    Nearest to during: after, in, through, before, until, throughout, when, following,
    Nearest to six: five, seven, eight, four, three, nine, zero, two,
    Nearest to state: city, states, government, prince, loaders, biomes, ivy, provincial,
    Nearest to in: during, until, on, within, into, throughout, pew, after,
    Nearest to over: around, explodes, dodo, poseidon, aleutians, into, measuring, about,
    Nearest to i: ii, we, you, g, they, cii, iii, t,
    Nearest to known: used, defined, regarded, classified, seen, such, considered, called,
    Average loss at step 82000: 3.410091
    Average loss at step 84000: 3.408310
    Average loss at step 86000: 3.392920
    Average loss at step 88000: 3.350469
    Average loss at step 90000: 3.363031
    Nearest to first: second, next, last, third, truman, same, halting, sailor,
    Nearest to when: while, before, if, after, where, though, although, until,
    Nearest to is: was, are, has, requires, becomes, does, although, makes,
    Nearest to their: its, his, her, our, your, the, them, some,
    Nearest to zero: five, seven, eight, six, four, nine, three, two,
    Nearest to new: slovakia, pirin, pacheco, projection, particular, renunciation, bmx, callous,
    Nearest to to: towards, circumscription, would, preoccupied, will, must, legged, safi,
    Nearest to or: and, glottal, unaspirated, nucleons, than, jacinto, though, ducati,
    Nearest to the: its, his, their, this, any, our, each, hurrians,
    Nearest to during: after, in, while, until, before, through, at, throughout,
    Nearest to six: seven, eight, five, nine, four, three, zero, two,
    Nearest to state: states, city, government, lysosomes, provincial, loaders, shin, malaya,
    Nearest to in: during, within, of, under, dhimmi, with, at, and,
    Nearest to over: around, poseidon, dodo, into, about, through, diem, measuring,
    Nearest to i: we, ii, you, g, they, cii, t, iii,
    Nearest to known: used, regarded, defined, such, classified, seen, described, referred,
    Average loss at step 92000: 3.396992
    Average loss at step 94000: 3.256399
    Average loss at step 96000: 3.356999
    Average loss at step 98000: 3.241581
    Average loss at step 100000: 3.355274
    Nearest to first: next, last, second, third, truman, halting, defeat, sailor,
    Nearest to when: if, while, where, after, before, though, although, during,
    Nearest to is: was, has, are, became, be, syringe, requires, had,
    Nearest to their: its, his, her, our, your, the, some, these,
    Nearest to zero: five, seven, eight, six, four, nine, three, two,
    Nearest to new: projection, slovakia, monkeys, chancery, renunciation, callous, pacheco, conceiving,
    Nearest to to: will, must, could, and, would, should, towards, can,
    Nearest to or: and, glottal, than, vg, transitioned, optic, whedon, dawson,
    Nearest to the: their, his, its, a, whose, your, her, some,
    Nearest to during: after, in, before, through, throughout, until, following, within,
    Nearest to six: seven, eight, four, five, nine, three, two, zero,
    Nearest to state: states, city, provincial, government, prince, shin, enumerable, lysosomes,
    Nearest to in: during, within, on, until, at, merdeka, from, dhimmi,
    Nearest to over: around, routine, through, within, full, about, measuring, dodo,
    Nearest to i: we, you, ii, they, he, rip, g, t,
    Nearest to known: defined, such, used, regarded, classified, possible, described, seen,



```python
num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])
```


```python
def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  pylab.show()

words = [reverse_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)
```

---

Problem
-------

An alternative to skip-gram is another Word2Vec model called [CBOW](http://arxiv.org/abs/1301.3781) (Continuous Bag of Words). In the CBOW model, instead of predicting a context word from a word vector, you predict a word from the sum of all the word vectors in its context. Implement and evaluate a CBOW model trained on the text8 dataset.

---

The difference between **Skip-Gram model** and the **Continuous Bag-of-Words (CBOW)** model is that while **Skip-Gram model** predicts context words from the target words, **CBOW** model does the inverse and predicts target words from context words. *[From Tensorflow.org]*

In order to use the CBOW model, I will update the generate_batch function so that **batch** contains context words and **labels** contain target words.


```python
data_index = 0

def generate_batch_cbow(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      
      # Everything stays the same as the Skip-Gram model except
      # labels and batch are switched
      labels[i * num_skips + j, 0] = buffer[skip_window]
      batch[i * num_skips + j] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  
  return batch, labels

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch_cbow(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print(np.shape(batch))
    print(np.shape(labels))
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
```

    data: ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first']
    (8,)
    (8, 1)
    
    with num_skips = 2 and skip_window = 1:
        batch: ['as', 'anarchism', 'originated', 'a', 'as', 'term', 'a', 'of']
        labels: ['originated', 'originated', 'as', 'as', 'a', 'a', 'term', 'term']
    (8,)
    (8, 1)
    
    with num_skips = 4 and skip_window = 2:
        batch: ['originated', 'anarchism', 'term', 'a', 'of', 'as', 'originated', 'term']
        labels: ['as', 'as', 'as', 'as', 'a', 'a', 'a', 'a']



```python
batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. 
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):

  # Input data.
  train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  
  # Variables.
  embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  softmax_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                         stddev=1.0 / math.sqrt(embedding_size)))
  softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Model.
  # Look up embeddings for inputs.
  embed = tf.nn.embedding_lookup(embeddings, train_dataset)
  # Compute the softmax loss, using a sample of the negative labels each time.
  loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                               train_labels, num_sampled, vocabulary_size))

  # Optimizer.
  optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
  
  # Compute the similarity between minibatch examples and all embeddings.
  # We use the cosine distance:
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
```


```python
num_steps = 100001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  average_loss = 0
  for step in range(num_steps):
    batch_data, batch_labels = generate_batch_cbow(
      batch_size, num_skips, skip_window)
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += l
    if step % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step %d: %f' % (step, average_loss))
      average_loss = 0
    # note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log = '%s %s,' % (log, close_word)
        print(log)
  final_embeddings = normalized_embeddings.eval()
```

    Initialized
    Average loss at step 0: 8.044578
    Nearest to who: shrunk, civilized, jeroboam, ulvaeus, monarchic, excerpt, maroons, microbes,
    Nearest to will: hq, concedes, stipend, garland, turing, elucidation, renaissance, outer,
    Nearest to d: continual, disenfranchisement, azul, mazar, convictions, abundantly, capoeiristas, multicellular,
    Nearest to after: sociable, ww, abolishing, devious, remainder, lester, secularization, cartridge,
    Nearest to have: afl, flattening, couscous, tightening, synonymous, adrenaline, understood, stuntman,
    Nearest to american: hint, polemic, solemnly, sages, pasha, dislikes, eb, birka,
    Nearest to between: currency, venomous, pcc, pcbs, versatility, cantons, indulged, premium,
    Nearest to use: imre, vivekananda, convicts, jass, stairs, allophones, urinating, geller,
    Nearest to at: tot, hing, esdi, each, clinics, laugh, futile, immemorial,
    Nearest to this: arabica, abandon, vav, echelon, devotion, hypoglycemic, functioned, polonaise,
    Nearest to he: nonprofit, ferdinand, oration, dsl, influenza, priests, ruskin, adversity,
    Nearest to only: academically, irate, endangered, aman, notre, epidemics, monarchies, canis,
    Nearest to six: loyalty, icsu, coeur, inspections, forecast, advancing, destabilization, tum,
    Nearest to in: sunflower, silvanus, friday, magic, enterprise, baptized, discipleship, stoicism,
    Nearest to united: galway, posited, petty, summit, consonant, vesuvius, sandworm, julia,
    Nearest to called: connoisseur, dump, blood, si, bars, guilds, fung, monarchists,
    Average loss at step 2000: 4.368932
    Average loss at step 4000: 3.866541
    Average loss at step 6000: 3.792614
    Average loss at step 8000: 3.682499
    Average loss at step 10000: 3.613551
    Nearest to who: he, they, monarchic, and, she, ecuador, csf, civilized,
    Nearest to will: would, earley, etext, daugavpils, concedes, can, exilic, hq,
    Nearest to d: continual, sindhu, and, abundantly, denounced, koenigsegg, ordaining, bricks,
    Nearest to after: homeomorphism, asis, learns, charging, dialectal, apo, secularization, cartridge,
    Nearest to have: be, had, has, were, are, adrenaline, do, dirt,
    Nearest to american: ean, resorts, and, sages, undisputed, crusade, containing, solemnly,
    Nearest to between: ocp, grasped, freeza, indulged, forties, currency, pcc, carousel,
    Nearest to use: imre, vivekananda, development, shiva, geller, sublime, meriwether, ozzfest,
    Nearest to at: during, tot, in, by, on, hing, from, clinics,
    Nearest to this: it, the, which, some, ithaca, norah, worldstatesmen, abandon,
    Nearest to he: it, she, they, who, there, loudspeaker, hdi, icbm,
    Nearest to only: iss, endangered, academically, notre, anthropic, elision, household, then,
    Nearest to six: eight, seven, three, five, four, nine, zero, two,
    Nearest to in: of, from, on, with, for, at, dyer, among,
    Nearest to united: galway, posited, fugitive, last, genji, dishes, enlargement, consonant,
    Nearest to called: monad, ghola, guilds, bars, nolan, promenade, erie, travolta,
    Average loss at step 12000: 3.605517
    Average loss at step 14000: 3.570433
    Average loss at step 16000: 3.404587
    Average loss at step 18000: 3.458411
    Average loss at step 20000: 3.539390
    Nearest to who: he, which, she, they, also, never, often, csf,
    Nearest to will: would, can, could, should, may, to, exilic, might,
    Nearest to d: b, denounced, hamburger, conditioner, sindhu, notated, bricks, shooter,
    Nearest to after: before, asis, homeomorphism, learns, dialectal, dail, by, against,
    Nearest to have: had, has, be, are, were, having, inconceivable, shinto,
    Nearest to american: english, crusade, resorts, polemic, hobsbawm, sages, layering, solemnly,
    Nearest to between: with, around, ocp, spindle, indulged, playa, from, infested,
    Nearest to use: imre, vivekananda, part, shiva, sublime, development, harps, malacca,
    Nearest to at: in, on, during, tot, from, with, under, ovarian,
    Nearest to this: it, which, the, some, norah, apocrypha, his, devotion,
    Nearest to he: it, she, they, who, there, loudspeaker, but, hdi,
    Nearest to only: iss, irate, anthropic, elision, even, spoons, endangered, notre,
    Nearest to six: eight, seven, four, five, nine, three, two, zero,
    Nearest to in: at, on, with, from, and, pitot, during, among,
    Nearest to united: genji, fugitive, dishes, galway, posited, affix, enlargement, last,
    Nearest to called: mel, ghola, monad, bars, travolta, nolan, irishman, guilds,
    Average loss at step 22000: 3.502293
    Average loss at step 24000: 3.488388
    Average loss at step 26000: 3.479927
    Average loss at step 28000: 3.477041
    Average loss at step 30000: 3.503812
    Nearest to who: he, they, she, which, never, also, many, it,
    Nearest to will: would, can, may, could, should, exilic, might, to,
    Nearest to d: b, hamburger, shooter, enhancing, sindhu, conditioner, geneticist, quad,
    Nearest to after: before, when, during, in, against, astrometric, from, asis,
    Nearest to have: had, has, were, having, are, be, include, make,
    Nearest to american: english, russian, british, french, sages, crusade, doll, hobsbawm,
    Nearest to between: with, around, in, from, infested, ocp, playa, workplaces,
    Nearest to use: imre, vivekananda, development, shiva, retardation, sublime, mung, stand,
    Nearest to at: in, during, screen, under, on, tot, hon, censorial,
    Nearest to this: it, which, that, the, lpc, he, some, there,
    Nearest to he: she, it, they, who, there, hdi, ushering, fluorite,
    Nearest to only: not, endangered, civilized, proposed, present, incorrect, even, warden,
    Nearest to six: eight, four, five, seven, nine, three, zero, two,
    Nearest to in: at, during, until, of, draped, on, and, around,
    Nearest to united: fugitive, genji, enlargement, affix, kut, posited, galway, gerais,
    Nearest to called: ghola, mel, travolta, monad, billiards, nationalized, nolan, deduction,
    Average loss at step 32000: 3.499731
    Average loss at step 34000: 3.495307
    Average loss at step 36000: 3.452651
    Average loss at step 38000: 3.296763
    Average loss at step 40000: 3.421513
    Nearest to who: he, which, she, also, they, often, never, generally,
    Nearest to will: would, can, could, should, may, must, might, cannot,
    Nearest to d: b, UNK, hamburger, hubs, tiles, adv, lyricists, quad,
    Nearest to after: before, during, when, astrometric, asis, dialectal, turntables, against,
    Nearest to have: had, has, were, be, having, are, include, waterfowl,
    Nearest to american: british, english, crusade, german, french, russian, sages, solemnly,
    Nearest to between: with, around, from, ocp, infested, fortuitous, abiding, kj,
    Nearest to use: imre, development, shiva, reach, retardation, vivekananda, support, stand,
    Nearest to at: on, in, during, tot, bosniak, duplicated, album, for,
    Nearest to this: which, the, it, that, any, some, another, advising,
    Nearest to he: she, it, they, who, there, i, hdi, then,
    Nearest to only: not, notre, even, survived, endangered, naturally, hopeless, officially,
    Nearest to six: four, five, eight, seven, nine, three, two, one,
    Nearest to in: during, at, of, dyer, for, from, on, until,
    Nearest to united: fugitive, kut, genji, affix, enlargement, aegean, west, posited,
    Nearest to called: mel, nationalized, ghola, sailors, travolta, nolan, irishman, considered,
    Average loss at step 42000: 3.432169
    Average loss at step 44000: 3.451831
    Average loss at step 46000: 3.451181
    Average loss at step 48000: 3.350450
    Average loss at step 50000: 3.383377
    Nearest to who: he, she, which, they, never, also, interdependence, generally,
    Nearest to will: would, could, can, should, may, must, might, cannot,
    Nearest to d: b, m, l, hamburger, kosovar, t, o, sindhu,
    Nearest to after: before, when, during, while, turntables, afterwards, dialectal, since,
    Nearest to have: had, has, were, are, be, having, spectacularly, refer,
    Nearest to american: english, crusade, french, african, british, russian, sages, australian,
    Nearest to between: with, around, among, from, in, ocp, fortuitous, spindle,
    Nearest to use: imre, development, form, retardation, end, part, share, amount,
    Nearest to at: during, tot, in, on, cutaway, ovarian, from, monopoles,
    Nearest to this: which, it, the, he, novella, deadlines, oscillations, some,
    Nearest to he: she, it, they, who, there, hdi, this, eventually,
    Nearest to only: first, officially, certainly, iss, endangered, proposed, draper, naturally,
    Nearest to six: eight, seven, four, five, three, nine, two, zero,
    Nearest to in: during, within, on, from, at, until, of, between,
    Nearest to united: fugitive, genji, kut, affix, enlargement, aegean, mosley, posited,
    Nearest to called: mel, known, travolta, used, nationalized, knitted, dump, nolan,
    Average loss at step 52000: 3.439855
    Average loss at step 54000: 3.423912
    Average loss at step 56000: 3.438553
    Average loss at step 58000: 3.396980
    Average loss at step 60000: 3.390309
    Nearest to who: he, she, never, often, which, they, generally, also,
    Nearest to will: would, could, can, may, should, must, might, cannot,
    Nearest to d: b, t, m, quad, l, kosovar, c, hubs,
    Nearest to after: before, during, when, while, until, despite, afterwards, astrometric,
    Nearest to have: had, has, were, having, are, be, spectacularly, clogged,
    Nearest to american: british, crusade, english, russian, african, international, german, french,
    Nearest to between: with, among, around, in, from, infested, spindle, ocp,
    Nearest to use: imre, form, share, retardation, hor, reflective, reach, respect,
    Nearest to at: during, tot, in, on, ovarian, hing, immemorial, fortuitous,
    Nearest to this: which, it, some, there, that, the, websites, he,
    Nearest to he: she, it, they, there, who, we, pulverized, this,
    Nearest to only: quenched, officially, first, replicating, but, probably, interlinked, rca,
    Nearest to six: eight, four, five, nine, seven, three, two, zero,
    Nearest to in: within, during, since, on, at, of, with, until,
    Nearest to united: fugitive, kut, genji, enlargement, aegean, posited, affix, refractory,
    Nearest to called: used, considered, known, knitted, mel, referred, stigma, spassky,
    Average loss at step 62000: 3.240127
    Average loss at step 64000: 3.257526
    Average loss at step 66000: 3.403863
    Average loss at step 68000: 3.396707
    Average loss at step 70000: 3.356754
    Nearest to who: he, she, never, they, richard, often, meine, bello,
    Nearest to will: would, could, may, can, should, must, might, cannot,
    Nearest to d: b, quad, sindhu, l, g, sar, m, adv,
    Nearest to after: before, during, when, while, until, against, sensible, stephanie,
    Nearest to have: had, has, were, are, be, having, include, contain,
    Nearest to american: english, british, african, industrial, seuss, german, international, predatory,
    Nearest to between: with, among, around, from, in, within, denarius, anasazi,
    Nearest to use: share, form, reach, retardation, imre, amount, reflective, hor,
    Nearest to at: during, tot, bosniak, zonker, album, releasing, in, cutaway,
    Nearest to this: which, it, the, some, another, that, there, websites,
    Nearest to he: she, it, they, there, who, we, hdi, then,
    Nearest to only: easily, officially, interlinked, quenched, probable, still, changes, probably,
    Nearest to six: eight, seven, nine, four, five, three, two, zero,
    Nearest to in: within, during, on, for, from, of, until, through,
    Nearest to united: fugitive, aegean, genji, kut, himachal, arctic, kentucky, mosley,
    Nearest to called: considered, named, known, referred, used, knitted, nationalized, tachycardia,
    Average loss at step 72000: 3.371856
    Average loss at step 74000: 3.351158
    Average loss at step 76000: 3.310976
    Average loss at step 78000: 3.352848
    Average loss at step 80000: 3.379728
    Nearest to who: he, she, never, they, bello, which, often, moriarty,
    Nearest to will: would, can, could, may, should, must, might, cannot,
    Nearest to d: b, g, m, sindhu, hubs, l, notated, p,
    Nearest to after: before, during, when, despite, while, until, without, from,
    Nearest to have: had, has, are, be, having, include, were, refer,
    Nearest to american: british, english, german, seuss, french, african, international, association,
    Nearest to between: among, within, with, around, fortuitous, from, flea, denarius,
    Nearest to use: form, share, because, reach, way, retardation, imre, reflective,
    Nearest to at: during, in, tot, releasing, despite, duplicated, bahr, calories,
    Nearest to this: which, it, the, he, some, norah, itself, another,
    Nearest to he: she, it, they, there, who, we, this, never,
    Nearest to only: twisting, weezer, quenched, officially, probably, notre, actually, easily,
    Nearest to six: five, seven, four, eight, three, nine, two, zero,
    Nearest to in: during, on, within, draped, until, since, at, from,
    Nearest to united: fugitive, genji, arctic, aegean, confederation, himachal, kentucky, gerais,
    Nearest to called: known, considered, used, referred, named, lleida, knitted, deduction,
    Average loss at step 82000: 3.404964
    Average loss at step 84000: 3.411332
    Average loss at step 86000: 3.388920
    Average loss at step 88000: 3.349558
    Average loss at step 90000: 3.365286
    Nearest to who: he, she, often, they, which, richard, never, angevin,
    Nearest to will: would, can, could, may, must, should, might, cannot,
    Nearest to d: b, sindhu, m, g, t, notated, l, convolutions,
    Nearest to after: before, during, when, while, despite, until, for, without,
    Nearest to have: had, has, were, having, are, be, include, contain,
    Nearest to american: british, french, english, african, italian, european, japanese, german,
    Nearest to between: with, within, around, among, in, fortuitous, flea, denarius,
    Nearest to use: share, most, respect, medlineplus, because, development, reflective, epicurean,
    Nearest to at: during, in, tot, rump, near, zirconium, gabriel, after,
    Nearest to this: which, it, another, some, renunciation, itself, the, he,
    Nearest to he: she, it, they, there, who, later, we, but,
    Nearest to only: weezer, actually, interlinked, titans, sterling, trusts, thebes, craftsmen,
    Nearest to six: eight, seven, four, five, nine, three, zero, two,
    Nearest to in: during, within, on, of, under, at, between, and,
    Nearest to united: fugitive, genji, arctic, aegean, confederation, rest, himachal, rust,
    Nearest to called: considered, known, used, named, referred, nolan, termed, nationalized,
    Average loss at step 92000: 3.397223
    Average loss at step 94000: 3.254230
    Average loss at step 96000: 3.357225
    Average loss at step 98000: 3.241184
    Average loss at step 100000: 3.356200
    Nearest to who: he, she, often, never, which, they, that, richard,
    Nearest to will: would, can, could, may, should, must, might, cannot,
    Nearest to d: b, quad, sindhu, p, c, genomes, g, lyricists,
    Nearest to after: before, during, when, despite, without, while, though, until,
    Nearest to have: had, has, are, contain, be, having, include, were,
    Nearest to american: british, french, russian, italian, english, canadian, german, japanese,
    Nearest to between: with, within, among, around, infested, flea, fortuitous, following,
    Nearest to use: share, respect, medlineplus, most, form, retardation, reflective, antithesis,
    Nearest to at: during, tines, on, tot, near, decius, in, duplicated,
    Nearest to this: which, it, another, itself, some, the, that, what,
    Nearest to he: she, it, they, we, there, who, larousse, never,
    Nearest to only: interlinked, craftsmen, weezer, quenched, actually, titans, naturally, cornered,
    Nearest to six: seven, four, five, eight, nine, three, two, zero,
    Nearest to in: during, within, on, among, until, throughout, from, near,
    Nearest to united: fugitive, genji, arctic, confederate, aegean, communist, himachal, confederation,
    Nearest to called: known, named, considered, referred, termed, used, nationalized, lleida,



```python
num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])
```


```python
def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  pylab.show()

words = [reverse_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)
```
