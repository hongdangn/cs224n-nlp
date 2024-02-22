import sys

class PartialParse(object):
    def __init__(self, sentence):
        self.sentence = sentence
        self.stack = ['ROOT']
        self.buffer = sentence
        self.dependencies = []

    def parse_step(self, transition):
        if transition == "S":
          if len(self.buffer) != 0:
           self.stack.append(self.buffer[0])
           self.buffer = [self.buffer[id] for id in range(len(self.buffer)) if id != 0]
        elif transition == "LA":
          self.dependencies.append((self.stack[-1], self.stack[-2]))
          self.stack = [self.stack[id] for id in range(len(self.stack)) if id != len(self.stack) - 2]
        else:
          self.dependencies.append((self.stack[-2], self.stack[-1]))
          self.stack = [self.stack[id] for id in range(len(self.stack)) if id != len(self.stack) - 1]

    def parse(self, transitions):
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies

def minibatch_parse(sentences, model, batch_size):
    dependencies = []
    partial_parses = [PartialParse(sentence) for sentence in sentences]
    unfinished_parses = partial_parses

    while len(unfinished_parses) != 0:
      batch_parses = unfinished_parses[: batch_size]
      transitions = model.predict(batch_parses)

      for id in range(len(batch_parses)):
        batch_parses[id].parse_step(transitions[id])
        if len(batch_parses[id].stack) == 1 and len(batch_parses[id].buffer) == 0:
          dependency = batch_parses[id].dependencies
          dependencies.append(dependency)
          unfinished_parses = [unfinished_parses[i] for i in range(len(unfinished_parses)) if i != id]
          
    return dependencies

class DummyModel(object):
    """Dummy model for testing the minibatch_parse function
    """
    def __init__(self, mode = "unidirectional"):
        self.mode = mode

    def predict(self, partial_parses):
        if self.mode == "unidirectional":
            return self.unidirectional_predict(partial_parses)
        elif self.mode == "interleave":
            return self.interleave_predict(partial_parses)
        else:
            raise NotImplementedError()

    def unidirectional_predict(self, partial_parses):
        """First shifts everything onto the stack and then does exclusively right arcs if the first word of
        the sentence is "right", "left" if otherwise.
        """
        return [("RA" if pp.stack[1] == "right" else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]

    def interleave_predict(self, partial_parses):
        """First shifts everything onto the stack and then interleaves "right" and "left".
        """
        return [("RA" if len(pp.stack) % 2 == 0 else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]

def test_dependencies(name, deps, ex_deps):
    """Tests the provided dependencies match the expected dependencies"""
    deps = tuple(sorted(deps))
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)

# Unidirectional arcs test
sentences = [["right", "arcs", "only"],
              ["right", "arcs", "only", "again"],
              ["left", "arcs", "only"],
              ["left", "arcs", "only", "again"]]
deps = minibatch_parse(sentences, DummyModel(), 2)
test_dependencies("minibatch_parse", deps[0],
                  (('ROOT', 'right'), ('arcs', 'only'), ('right', 'arcs')))
test_dependencies("minibatch_parse", deps[1],
                  (('ROOT', 'right'), ('arcs', 'only'), ('only', 'again'), ('right', 'arcs')))
test_dependencies("minibatch_parse", deps[2],
                  (('only', 'ROOT'), ('only', 'arcs'), ('only', 'left')))
test_dependencies("minibatch_parse", deps[3],
                  (('again', 'ROOT'), ('again', 'arcs'), ('again', 'left'), ('again', 'only')))

# Out-of-bound test
sentences = [["right"]]
deps = minibatch_parse(sentences, DummyModel(), 2)
test_dependencies("minibatch_parse", deps[0], (('ROOT', 'right'),))

# Mixed arcs test
sentences = [["this", "is", "interleaving", "dependency", "test"]]
deps = minibatch_parse(sentences, DummyModel(mode="interleave"), 1)
test_dependencies("minibatch_parse", deps[0],
                  (('ROOT', 'is'), ('dependency', 'interleaving'),
                  ('dependency', 'test'), ('is', 'dependency'), ('is', 'this')))
print("minibatch_parse test passed!")