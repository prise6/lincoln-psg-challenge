# -*- coding: utf-8 -*-

import yaml
import os
from dotenv import find_dotenv, load_dotenv
from src.tools import Config

def main():

	cfg = Config(project_dir = os.getenv('PROJECT_DIR'), mode = os.getenv('MODE'))
	replace_items_recursive(cfg.config)
	print(cfg.project_dir + '/config/config.template.yaml')
	with open(cfg.project_dir + '/config/config.template.yaml', 'w') as f:
		yaml.dump(cfg.config, f, default_flow_style = False)

def replace_items_recursive(d, v = 'XXX'):
    for k in d.keys():
        if type(d.get(k)) is not dict:
            d.update({k: v})
        else:
            replace_items_recursive(d.get(k))

if __name__ == '__main__':
	load_dotenv(find_dotenv())
	main()


