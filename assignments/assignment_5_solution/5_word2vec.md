
Deep Learning
=============

Assignment 5
------------

The goal of this assignment is to train a Word2Vec skip-gram model over [Text8](http://mattmahoney.net/dc/textdata) data.


```python
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
%matplotlib inline
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
print(words[:40])
```

    Data size 17005207
    ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against', 'early', 'working', 'class', 'radicals', 'including', 'the', 'diggers', 'of', 'the', 'english', 'revolution', 'and', 'the', 'sans', 'culottes', 'of', 'the', 'french', 'revolution', 'whilst', 'the', 'term', 'is', 'still', 'used', 'in', 'a', 'pejorative', 'way', 'to']


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
    Sample data [5241, 3082, 12, 6, 195, 2, 3136, 46, 59, 156]


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
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
```

    data: ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first']
    
    with num_skips = 2 and skip_window = 1:
        batch: ['originated', 'originated', 'as', 'as', 'a', 'a', 'term', 'term']
        labels: ['as', 'anarchism', 'a', 'originated', 'as', 'term', 'a', 'of']
    
    with num_skips = 4 and skip_window = 2:
        batch: ['as', 'as', 'as', 'as', 'a', 'a', 'a', 'a']
        labels: ['anarchism', 'a', 'term', 'originated', 'term', 'as', 'originated', 'of']


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
  # Note: The optimizer will optimize the softmax_weights AND the embeddings.
  # This is because the embeddings are defined as a variable quantity and the
  # optimizer's `minimize` method will by default modify all variable quantities 
  # that contribute to the tensor it is passed.
  # See docs on `tf.train.Optimizer.minimize()` for more details.
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
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log = '%s %s,' % (log, close_word)
        print(log)
  final_embeddings = normalized_embeddings.eval()
```

    Initialized
    Average loss at step 0: 7.865581
    Nearest to such: bloemfontein, cassell, dicis, articulations, arabized, skipping, syd, origami,
    Nearest to had: shortcuts, permitted, congregate, snopes, ruth, aerosmith, adoption, vickers,
    Nearest to new: rif, march, bladed, priesthood, egoism, locksmithing, rican, moorcock,
    Nearest to nine: derivations, eeeeee, gomorrah, mock, hui, iic, louth, detritus,
    Nearest to first: ubiquity, orthographies, yourdon, oversees, mac, flickering, motion, faber,
    Nearest to with: cur, cryptography, socket, jdk, mcgovern, donegal, copernican, deeds,
    Nearest to will: laurie, decode, carboxyl, cartel, cisc, fontainebleau, phosphates, cherry,
    Nearest to have: stigma, crooks, dose, swann, tone, cheyenne, shan, hydrolyzed,
    Nearest to in: pancake, exhaust, airfoil, antimatter, lagrange, amplifies, grateful, masoretic,
    Nearest to war: subway, disproportionate, incarnate, ish, lichen, hark, giambattista, bail,
    Nearest to at: vigil, pals, fever, beers, lavender, poole, nutritious, monitoring,
    Nearest to system: professions, antiprotons, sacd, langle, matter, mafia, detailing, notified,
    Nearest to use: stronghold, worsen, mifflin, lichfield, goo, customers, lcc, sculptors,
    Nearest to no: violent, sera, injustice, dih, additive, minted, confrontation, ron,
    Nearest to for: humorous, urges, bijective, rorty, subjective, roadways, level, norad,
    Nearest to other: handball, bittersweet, acacias, bsod, beltane, catalina, glossolalia, asgard,
    Average loss at step 2000: 4.357096
    Average loss at step 4000: 3.862722
    Average loss at step 6000: 3.777233
    Average loss at step 8000: 3.670959
    Average loss at step 10000: 3.646112
    Nearest to such: known, long, well, bloemfontein, tiryns, origami, netsplit, articulations,
    Nearest to had: has, was, have, been, vickers, letterbox, isotropic, would,
    Nearest to new: rif, seasonal, march, christian, witbrock, pistons, volo, egoism,
    Nearest to nine: eight, seven, five, four, six, zero, three, one,
    Nearest to first: flickering, fernando, gonzo, ubiquity, zona, smugglers, pelletier, proportional,
    Nearest to with: between, mcgovern, as, in, for, blob, consented, mimics,
    Nearest to will: laurie, begin, depose, lerner, cancers, to, krakatau, litre,
    Nearest to have: be, has, had, are, academy, hydrolyzed, encyclopedists, handicaps,
    Nearest to in: on, at, from, by, of, during, with, after,
    Nearest to war: disproportionate, throughput, chamber, subway, incarnate, bx, inanimate, bail,
    Nearest to at: in, on, deceiver, after, gomoku, imp, pulpit, by,
    Nearest to system: mafia, encampments, pattern, matter, healthier, sip, sony, sahih,
    Nearest to use: stronghold, worsen, diagrams, optimates, eriksson, peaceful, intoxicated, anders,
    Nearest to no: confrontation, crafty, prana, above, anthemius, cremation, tampere, knobs,
    Nearest to for: with, as, in, best, systematized, at, subjective, makeup,
    Nearest to other: sights, catalina, candela, handball, bsod, cent, beltane, terrific,
    Average loss at step 12000: 3.620532
    Average loss at step 14000: 3.475965
    Average loss at step 16000: 3.427487
    Average loss at step 18000: 3.554333
    Average loss at step 20000: 3.507307
    Nearest to such: known, well, netsplit, mathew, these, bloemfontein, tiryns, cremona,
    Nearest to had: has, have, was, took, were, been, letterbox, isotropic,
    Nearest to new: pistons, rif, greek, seasonal, christian, slade, gustaf, march,
    Nearest to nine: eight, seven, three, six, five, four, zero, one,
    Nearest to first: second, last, flickering, gonzo, next, reactor, reconstructionist, altering,
    Nearest to with: between, against, distressed, for, consented, in, by, to,
    Nearest to will: would, can, may, might, could, krakatau, lerner, should,
    Nearest to have: had, has, be, were, hold, kinetochores, are, encyclopedists,
    Nearest to in: at, on, of, from, during, with, through, thorne,
    Nearest to war: throughput, disproportionate, bx, planck, bail, incarnate, second, chamber,
    Nearest to at: in, on, for, near, swaziland, after, remaking, dieu,
    Nearest to system: mafia, matter, revisionists, sacd, tens, sony, thera, antiprotons,
    Nearest to use: stronghold, worsen, eriksson, implementations, form, thickened, rightarrow, intoxicated,
    Nearest to no: confrontation, prana, anthemius, any, tampere, knobs, suits, additive,
    Nearest to for: at, with, to, of, socialize, on, readability, and,
    Nearest to other: candela, these, many, different, various, asch, some, cent,
    Average loss at step 22000: 3.486945
    Average loss at step 24000: 3.480552
    Average loss at step 26000: 3.521966
    Average loss at step 28000: 3.465381
    Average loss at step 30000: 3.532980
    Nearest to such: known, well, these, netsplit, many, mathew, kronstadt, origami,
    Nearest to had: has, have, was, were, since, took, gave, been,
    Nearest to new: rif, seasonal, deuteronomy, christian, slade, pistons, physical, gustaf,
    Nearest to nine: eight, seven, six, four, three, five, one, zero,
    Nearest to first: last, second, next, same, shelf, restrained, zona, flickering,
    Nearest to with: between, against, rin, without, consented, in, onstage, bcl,
    Nearest to will: can, would, may, might, could, should, must, cannot,
    Nearest to have: had, has, are, be, were, hold, represent, handicaps,
    Nearest to in: at, during, and, on, hornets, from, under, burrow,
    Nearest to war: throughput, bx, disproportionate, incarnate, ece, second, planck, inanimate,
    Nearest to at: in, during, on, after, throughout, gomoku, high, near,
    Nearest to system: systems, revisionists, tens, nieve, mafia, sinti, hennig, cryptographer,
    Nearest to use: worsen, stronghold, weill, form, eriksson, implementations, bukharin, intoxicated,
    Nearest to no: confrontation, any, anthemius, prana, cornet, guang, already, ibu,
    Nearest to for: without, when, after, against, shinogi, teleplay, wt, nuh,
    Nearest to other: various, different, many, these, candela, several, mantle, asch,
    Average loss at step 32000: 3.467085
    Average loss at step 34000: 3.493512
    Average loss at step 36000: 3.267101
    Average loss at step 38000: 3.485309
    Average loss at step 40000: 3.428583
    Nearest to such: known, these, well, many, netsplit, mathew, kronstadt, including,
    Nearest to had: has, have, were, was, would, since, gave, authorizing,
    Nearest to new: pistons, scoping, sheaf, slade, rif, deuteronomy, german, shoulders,
    Nearest to nine: eight, six, seven, five, four, three, zero, one,
    Nearest to first: last, second, next, shelf, restrained, heroes, reactor, monarchy,
    Nearest to with: between, against, rin, ding, onstage, ancestral, hawaiian, while,
    Nearest to will: can, would, may, could, should, might, must, to,
    Nearest to have: had, has, were, are, be, forcefully, handicaps, hold,
    Nearest to in: during, on, at, under, within, from, kelsey, hornets,
    Nearest to war: throughput, bx, disproportionate, chamber, planck, arsenic, jingle, exterminated,
    Nearest to at: during, on, within, in, after, gomoku, dieu, abrogation,
    Nearest to system: systems, tens, revisionists, mafia, adalbert, approximation, sony, pattern,
    Nearest to use: consider, form, worsen, bukharin, eriksson, weill, control, modest,
    Nearest to no: any, confrontation, prana, a, guang, already, anthemius, homininae,
    Nearest to for: socialize, best, with, subjective, nuh, sportswriter, mise, shinogi,
    Nearest to other: various, different, these, many, cent, mantle, candela, some,
    Average loss at step 42000: 3.448282
    Average loss at step 44000: 3.441201
    Average loss at step 46000: 3.396799
    Average loss at step 48000: 3.395908
    Average loss at step 50000: 3.394911
    Nearest to such: well, known, these, netsplit, mathew, many, bonnet, origami,
    Nearest to had: has, have, was, were, irreducibly, continued, led, been,
    Nearest to new: flatter, german, pistons, great, rif, slade, christian, deuteronomy,
    Nearest to nine: eight, seven, six, four, five, three, zero, two,
    Nearest to first: last, second, next, same, restrained, shelf, final, right,
    Nearest to with: between, cur, by, against, in, stratton, rin, ancestral,
    Nearest to will: would, can, could, may, must, might, should, shall,
    Nearest to have: had, has, be, were, are, hold, produce, choose,
    Nearest to in: during, at, on, since, from, until, within, massif,
    Nearest to war: throughput, bx, planck, jingle, contemplates, second, realpolitik, arsenic,
    Nearest to at: during, in, near, on, within, remaking, gomoku, versed,
    Nearest to system: systems, revisionists, tens, approximation, langle, adalbert, leda, sony,
    Nearest to use: consider, form, bukharin, regard, madhya, worsen, provocative, modest,
    Nearest to no: any, confrontation, opcode, prana, homininae, anthemius, ciliates, estas,
    Nearest to for: taller, while, against, banknotes, without, shinogi, including, teleplay,
    Nearest to other: various, many, different, mantle, asch, these, some, dad,
    Average loss at step 52000: 3.416827
    Average loss at step 54000: 3.443321
    Average loss at step 56000: 3.410834
    Average loss at step 58000: 3.426377
    Average loss at step 60000: 3.349711
    Nearest to such: known, well, these, netsplit, many, mathew, bonnet, regarded,
    Nearest to had: has, have, was, were, gave, been, having, led,
    Nearest to new: deuteronomy, recording, german, pistons, rif, flatter, slade, gustaf,
    Nearest to nine: eight, seven, six, five, four, three, one, two,
    Nearest to first: last, second, next, same, best, restrained, final, third,
    Nearest to with: between, including, distressed, ancestral, cur, while, piercer, mcgovern,
    Nearest to will: would, can, could, may, must, should, might, cannot,
    Nearest to have: had, has, are, were, produce, include, be, refer,
    Nearest to in: during, within, on, since, until, burrow, from, including,
    Nearest to war: throughput, bx, exterminated, realpolitik, geller, battle, rpr, jingle,
    Nearest to at: near, during, dieu, in, remaking, gomoku, throughout, within,
    Nearest to system: systems, network, tens, revisionists, dynamic, wilmut, approximation, mafia,
    Nearest to use: consider, bukharin, insecurity, form, move, practice, colchis, correspond,
    Nearest to no: any, prana, silva, cobb, homininae, ibu, little, kama,
    Nearest to for: after, including, shinogi, without, while, against, during, hays,
    Nearest to other: various, different, many, mantle, catalina, those, moabites, others,
    Average loss at step 62000: 3.099123
    Average loss at step 64000: 3.436185
    Average loss at step 66000: 3.369620
    Average loss at step 68000: 3.373482
    Average loss at step 70000: 3.406706
    Nearest to such: these, known, well, netsplit, many, certain, gentry, some,
    Nearest to had: have, has, was, were, authorizing, having, implausible, gave,
    Nearest to new: tycho, deuteronomy, rif, prominent, recording, carpets, slade, flatter,
    Nearest to nine: eight, six, seven, five, four, three, zero, one,
    Nearest to first: last, second, next, best, same, only, third, restrained,
    Nearest to with: between, against, cur, among, yapese, mcgovern, rin, ancestral,
    Nearest to will: would, could, can, may, must, should, might, shall,
    Nearest to have: had, has, were, be, are, hold, include, represent,
    Nearest to in: during, within, on, under, migne, since, watertown, fits,
    Nearest to war: throughput, battle, bx, exterminated, spacey, geller, contemplates, managing,
    Nearest to at: during, dieu, within, near, fastest, on, in, crypto,
    Nearest to system: systems, approximation, wilmut, process, revisionists, dynamic, mildly, junit,
    Nearest to use: consider, bukharin, olivia, cause, gatekeeper, feature, weill, regard,
    Nearest to no: any, little, sherry, homininae, there, cobb, prana, kama,
    Nearest to for: after, humorous, against, cannes, of, castel, submitting, computations,
    Nearest to other: various, different, others, some, mantle, many, historical, periodicity,
    Average loss at step 72000: 3.361806
    Average loss at step 74000: 3.273485
    Average loss at step 76000: 3.379170
    Average loss at step 78000: 3.355719
    Average loss at step 80000: 3.397946
    Nearest to such: well, known, these, netsplit, seen, regarded, logie, certain,
    Nearest to had: has, have, were, was, authorizing, implausible, having, suffered,
    Nearest to new: different, old, tycho, flatter, deuteronomy, nocs, hezbollah, zx,
    Nearest to nine: eight, seven, four, six, five, three, zero, two,
    Nearest to first: last, second, next, third, best, same, only, final,
    Nearest to with: between, against, cur, rin, gladiators, in, through, yapese,
    Nearest to will: would, could, can, may, might, should, must, cannot,
    Nearest to have: had, has, are, were, include, be, hold, having,
    Nearest to in: within, during, on, hornets, of, at, from, until,
    Nearest to war: bx, throughput, battle, exterminated, plasticity, geller, spacey, jingle,
    Nearest to at: during, in, around, within, near, dieu, gomoku, on,
    Nearest to system: systems, program, wilmut, network, approximation, revisionists, process, dynamic,
    Nearest to use: consider, cause, list, countercultural, regard, form, olivia, bukharin,
    Nearest to no: any, little, homininae, cobb, silva, already, reichenbach, intercourse,
    Nearest to for: while, against, peng, including, nuh, cannes, after, as,
    Nearest to other: various, different, others, including, catalina, many, several, periodicity,
    Average loss at step 82000: 3.419872
    Average loss at step 84000: 3.393403
    Average loss at step 86000: 3.358677
    Average loss at step 88000: 3.362531
    Average loss at step 90000: 3.379302
    Nearest to such: known, well, these, netsplit, logie, certain, follows, regarded,
    Nearest to had: has, have, was, were, having, since, would, authorizing,
    Nearest to new: different, deuteronomy, flatter, attentions, tycho, modern, marcellus, pastoral,
    Nearest to nine: seven, eight, six, five, four, three, zero, one,
    Nearest to first: last, second, next, third, best, only, final, same,
    Nearest to with: between, including, for, in, ancestral, akita, stratton, rin,
    Nearest to will: would, can, could, may, should, might, must, cannot,
    Nearest to have: had, has, include, be, hold, were, are, provide,
    Nearest to in: during, within, at, hornets, whispers, of, on, until,
    Nearest to war: battle, bx, throughput, geller, exterminated, ammonius, managing, jingle,
    Nearest to at: near, in, during, around, on, dieu, yan, after,
    Nearest to system: systems, process, wilmut, program, network, freshly, images, revisionists,
    Nearest to use: consider, cause, practice, regard, list, correspond, staircase, support,
    Nearest to no: any, little, already, mercenaries, hath, confrontation, silva, prana,
    Nearest to for: including, after, of, while, teleplay, by, with, without,
    Nearest to other: various, different, including, others, fretted, picked, older, humanism,
    Average loss at step 92000: 3.287597
    Average loss at step 94000: 3.333056
    Average loss at step 96000: 3.268359
    Average loss at step 98000: 3.340467
    Average loss at step 100000: 3.347028
    Nearest to such: known, well, these, logie, netsplit, follows, many, separate,
    Nearest to had: has, have, was, were, authorizing, would, since, did,
    Nearest to new: different, tycho, attentions, deuteronomy, gonna, flatter, fictional, datum,
    Nearest to nine: eight, seven, six, five, four, three, one, zero,
    Nearest to first: last, second, next, third, final, best, fourth, same,
    Nearest to with: between, inescapable, by, including, during, in, when, alveolar,
    Nearest to will: can, would, could, may, must, might, should, cannot,
    Nearest to have: had, has, were, be, authorizing, refer, include, are,
    Nearest to in: during, within, until, from, on, at, throughout, around,
    Nearest to war: battle, bx, throughput, rpr, managing, exterminated, ipsec, geller,
    Nearest to at: near, during, around, in, within, whyte, dieu, under,
    Nearest to system: systems, network, process, pattern, program, wilmut, campy, freshly,
    Nearest to use: cause, practice, support, bukharin, arouse, unaffected, mircea, correspond,
    Nearest to no: any, little, prana, already, mercenaries, yaw, liga, confrontation,
    Nearest to for: after, including, against, before, of, by, when, during,
    Nearest to other: various, moabites, humanism, others, older, several, harmonious, different,



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


![png](output_14_0.png)


---

Problem
-------

An alternative to skip-gram is another Word2Vec model called [CBOW](http://arxiv.org/abs/1301.3781) (Continuous Bag of Words). In the CBOW model, instead of predicting a context word from a word vector, you predict a word from the sum of all the word vectors in its context. Implement and evaluate a CBOW model trained on the text8 dataset.

---


```python
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):

    # Input data.
    train_dataset = tf.placeholder(
        tf.int32, shape=[batch_size, num_skips])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables.
    embeddings = tf.Variable(tf.random_uniform(
        [vocabulary_size, embedding_size], -1.0, 1.0))
    # Look up embeddings for inputs.
    embeds = tf.nn.embedding_lookup(embeddings, train_dataset)
    averaged_embed = tf.reduce_mean(embeds, 1)
    softmax_weights = tf.Variable(tf.truncated_normal(
        [vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.constant(.05, shape=[vocabulary_size]))

    # Model.
    # Compute the softmax loss, using a sample of the negative labels each
    # time.
    sampled_loss = tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, averaged_embed, train_labels, num_sampled, vocabulary_size)

    loss = tf.reduce_mean(sampled_loss)

    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's `minimize` method will by default modify all variable quantities
    # that contribute to the tensor it is passed.
    # See docs on `tf.train.Optimizer.minimize()` for more details.
    optimizer = tf.train.AdagradOptimizer(2.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, tf.transpose(normalized_embeddings))

```


```python
data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i, j] = buffer[target]
        labels[i] = buffer[skip_window]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


batch, labels = generate_batch(
    batch_size=8, num_skips=2, skip_window=1)
print('data:', [reverse_dictionary[di] for di in data[:8]])
print('batch:', [[reverse_dictionary[bi2] for bi2 in bi] for bi in batch])
print('labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

```

    data: ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first']
    batch: [['as', 'anarchism'], ['a', 'originated'], ['as', 'term'], ['of', 'a'], ['abuse', 'term'], ['of', 'first'], ['abuse', 'used'], ['first', 'against']]
    labels: ['originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used']



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

  sim = similarity.eval()
  for i in range(valid_size):
    valid_word = reverse_dictionary[valid_examples[i]]
    top_k = 8 # number of nearest neighbors
    nearest = (-sim[i, :]).argsort()[1:top_k+1]
    log = 'Nearest to %s:' % valid_word
    for k in range(top_k):
      close_word = reverse_dictionary[nearest[k]]
      log = '%s %s,' % (log, close_word)
    print(log)
  final_embeddings = normalized_embeddings.eval()
```

    Initialized
    Average loss at step 0: 7.846408
    Average loss at step 2000: 3.861637
    Average loss at step 4000: 3.410170
    Average loss at step 6000: 3.257011
    Average loss at step 8000: 3.125172
    Average loss at step 10000: 3.060920
    Average loss at step 12000: 3.085613
    Average loss at step 14000: 3.045867
    Average loss at step 16000: 3.066556
    Average loss at step 18000: 3.013001
    Average loss at step 20000: 2.887744
    Average loss at step 22000: 2.968505
    Average loss at step 24000: 2.931729
    Average loss at step 26000: 2.907379
    Average loss at step 28000: 2.932901
    Average loss at step 30000: 2.904251
    Average loss at step 32000: 2.740389
    Average loss at step 34000: 2.850137
    Average loss at step 36000: 2.846395
    Average loss at step 38000: 2.852099
    Average loss at step 40000: 2.836290
    Average loss at step 42000: 2.848900
    Average loss at step 44000: 2.870372
    Average loss at step 46000: 2.816769
    Average loss at step 48000: 2.780351
    Average loss at step 50000: 2.761199
    Average loss at step 52000: 2.801426
    Average loss at step 54000: 2.783858
    Average loss at step 56000: 2.771600
    Average loss at step 58000: 2.668390
    Average loss at step 60000: 2.741209
    Average loss at step 62000: 2.752960
    Average loss at step 64000: 2.695585
    Average loss at step 66000: 2.680049
    Average loss at step 68000: 2.637092
    Average loss at step 70000: 2.698919
    Average loss at step 72000: 2.719218
    Average loss at step 74000: 2.572752
    Average loss at step 76000: 2.722682
    Average loss at step 78000: 2.744865
    Average loss at step 80000: 2.704907
    Average loss at step 82000: 2.592513
    Average loss at step 84000: 2.690852
    Average loss at step 86000: 2.661496
    Average loss at step 88000: 2.673500
    Average loss at step 90000: 2.660424
    Average loss at step 92000: 2.607167
    Average loss at step 94000: 2.654514
    Average loss at step 96000: 2.632976
    Average loss at step 98000: 2.304476
    Average loss at step 100000: 2.338774
    Nearest to three: five, four, six, seven, eight, two, nine, zero,
    Nearest to i: you, we, ii, they, platypus, nozick, iii, iv,
    Nearest to a: immorality, gunpoint, undersecretary, transferable, the, wisegirls, caltech, annulled,
    Nearest to and: or, but, olwen, though, including, varanus, called, commonest,
    Nearest to was: is, seems, became, has, had, were, did, be,
    Nearest to no: little, any, judea, valentin, siddeley, prophylaxis, piccard, psc,
    Nearest to who: often, usually, she, actually, sometimes, kadyrov, deeply, urgell,
    Nearest to people: children, women, men, persons, individuals, residents, jews, americans,
    Nearest to about: regarding, around, approximately, whim, concerning, nominally, titus, modifying,
    Nearest to such: these, known, serve, well, many, follows, certain, served,
    Nearest to if: when, though, saw, before, did, unless, where, whenever,
    Nearest to were: are, have, include, was, contain, had, been, remain,
    Nearest to d: b, monistic, roanoke, pursues, talmadge, te, hemlock, outcrops,
    Nearest to than: or, kshatriya, much, least, considerably, nontrivial, sixty, numeral,
    Nearest to on: upon, through, canaris, within, onto, concerning, at, telemetry,
    Nearest to his: her, their, my, your, its, our, the, whose,

