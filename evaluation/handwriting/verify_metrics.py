"""Independent small-case metric checks; never executes a model."""
import random
from metrics import distance, score, visible


def slow(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        following = [i]
        for j, y in enumerate(b, 1):
            following.append(min(following[-1] + 1, row[j] + 1, row[j-1] + (x != y)))
        row = following
    return row[-1]


rng = random.Random(71)
for _ in range(250):
    a = ''.join(rng.choices('abc é١1', k=rng.randrange(30)))
    b = ''.join(rng.choices('abc é١1', k=rng.randrange(30)))
    assert distance(a, b) == slow(a, b)
    assert distance(a.split(), b.split()) == slow(a.split(), b.split())
assert visible('# A\n\n| B | C |\n| --- | --- |') == 'A B C'
assert score('١٢٣', '123')['cer'] == 100
assert score('é', 'e\u0301')['cer'] == 0
assert score('A', 'a')['cer'] == 100
assert score('word', 'word extra')['word_errors'] == 1
assert score('a', 'abc')['cer'] == 200
print('METRICS_VERIFIED 500 independent dynamic-programming comparisons; Unicode, digits, case, markup and insertion checks')
