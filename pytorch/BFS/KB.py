class KB(object):
	def __init__(self):
		self.entities = {}

	def addRelation(self, entity1, relation, entity2):
		if entity1 in self.entities:
			self.entities[entity1].append(Path(relation, entity2))
		else:
			self.entities[entity1] = [Path(relation, entity2)]

	def getPathsFrom(self, entity):
		return self.entities[entity]

	def removePath(self, entity1, entity2):
		for idx, path in enumerate(self.entities[entity1]):
			if(path.connected_entity == entity2):
				del self.entities[entity1][idx]
				break
		for idx, path in enumerate(self.entities[entity2]):
			if(path.connected_entity == entity1):
				del self.entities[entity2][idx]
				break

	def pickRandomIntermediatesBetween(self, entity1, entity2, num):
		"""
		pick random intermediate entities and return.
		Args:
			entity1: entity1
			entity2: entity2
			num: the number of intermediate entities
		"""
		# TO DO: COULD BE IMPROVED BY NARROWING THE RANGE OF RANDOM EACH TIME ITERATIVELY CHOOSE AN INTERMEDIATE
		# from sets import Set - set is built-in class in python 3
		import random

		res = set()
		if num > len(self.entities) - 2:
			raise ValueError('Number of Intermediates picked is larger than possible', 'num_entities: {}'.format(len(self.entities)), 'num_itermediates: {}'.format(num))
		for i in range(num):
			intermediate = random.choice(list(self.entities.keys()))
			while intermediate in res or intermediate == entity1 or intermediate == entity2:
				intermediate = random.choice(list(self.entities.keys()))
			res.add(intermediate)
		return list(res)

	def __str__(self):
		string = ""
		for entity in self.entities:
			string += entity + ','.join(str(x) for x in self.entities[entity])
			string += '\n'
		return string


class Path(object):
	def __init__(self, relation, connected_entity):
		self.relation = relation
		self.connected_entity = connected_entity

	def __str__(self):
		return "\t{}\t{}".format(self.relation, self.connected_entity)

	__repr__ = __str__
