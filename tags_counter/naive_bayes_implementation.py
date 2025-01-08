from collections import Counter, defaultdict
import pickle


# c = Counter()
# d = defaultdict(Counter)
# with open('world.class_from_names_with_tags.txt') as fr:
#     for l in fr:
#         temp = l.split(',', 1)
#         tags = [a.strip() for a in temp[1].split(',')]
#         d[temp[0].strip()] += Counter(tags)

# total_cat = {k: sum(d[k].values()) for k in d.keys()}
# total = sum(total_cat.values())

# with open('naive_bayes.pickle', 'wb') as f:
#     pickle.dump([d, total_cat, total], f)

with open('naive_bayes.pickle', 'rb') as f:
    d, total_cat, total = pickle.load(f)
print('restaurant: ', d['restaurant'].most_common(10), '\n\n',
      'bar: ', d['bar'].most_common(10), '\n\n',
      'hotel: ', d['hotel'].most_common(10), '\n\n',
      'cafe: ', d['cafe'].most_common(10))

tag = 'category_list_pizza_place'
probabilities = dict()
tag_total = sum(d[k1][tag] for k1 in d.keys()) + 1
prob_tag = tag_total / total
for k in d.keys():
    prob_cat = 1  # total_cat[k] / total  # throw away this if you want categories balanced (count of category does not matter)
    prob_tag_cat = d[k][tag] / tag_total
    probabilities[k] = prob_tag_cat * prob_cat / prob_tag
# notes: probability if tag belongs to category:  prob_cat is higher for e.g restaurant (higher frequency)
normalized = {k: v / max(probabilities.values()) for k, v in probabilities.items()}
print(normalized)
