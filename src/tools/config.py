# -*- coding: utf-8 -*-

import os
import sys
import yaml
import re

class Config:

	def __init__(self, project_dir, mode):
		
		self.project_dir = project_dir
		self.mode = mode
		self.path_matcher = re.compile(r'\$\{([^}^{]+)\}')

		yaml.add_implicit_resolver('!path', self.path_matcher, None, yaml.SafeLoader)
		yaml.add_constructor('!path', self.path_constructor, yaml.SafeLoader)

		with open(os.path.join(self.project_dir, 'config', 'config_%s.yaml' % (self.mode)), 'r') as ymlfile:
			self.config = yaml.safe_load(ymlfile)

	def get(self, key):
		return self.config[key]

	def path_constructor(self, loader, node):
		''' Extract the matched value, expand env variable, and replace the match '''
		value = node.value
		match = self.path_matcher.match(value)
		env_var = match.group()[2:-1]

		return os.environ.get(env_var) + value[match.end():]



