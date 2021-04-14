import sys

DEFAULT = 'glove_bot'

def run_bot(name):
	mod = __import__(name)
	mod.main()

if __name__ == '__main__':
	if len(sys.argv)==1:
		run_bot(DEFAULT)
	else:
		run_bot(sys.argv[1][2:])