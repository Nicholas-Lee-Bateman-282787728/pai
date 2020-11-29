import os
ROOT = os.path.dirname(os.path.abspath(__file__))
DIR_DATA = "{0}{1}data".format(ROOT, os.sep)
DIR_IMAGE = "{0}{1}image{1}".format(DIR_DATA, os.sep)
DIR_EMBEDDING = "{0}{1}embedding{1}".format(DIR_DATA, os.sep)
DIR_MODELS = "{0}models{1}".format(DIR_EMBEDDING, os.sep)
DIR_INPUT = "{0}{1}input{1}".format(DIR_DATA, os.sep)
DIR_OUTPUT = "{0}{1}output{1}".format(DIR_DATA, os.sep)
DIR_WMPSVAD = "{0}{1}model{1}".format(DIR_DATA, os.sep)
