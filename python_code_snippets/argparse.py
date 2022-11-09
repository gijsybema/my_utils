# documentation: https://docs.python.org/3/library/argparse.html

# specify list of options, this will be the only allowed options for the input. Will be used as value for argument choices in parser.add_argument()
list_options = ['option_1', 'option_2', 'option_3']

# define class to add ExtendAction to parser
class ExtendAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest) or []
        items.extend(values)
        setattr(namespace, self.dest, items)

# add extra action extend to argparse: 
parser = argparse.ArgumentParser()
parser.register('action', 'extend', ExtendAction)
parser.add_argument('input'
                    , action='extend'
                    , type=str                    
                    , nargs='+'
                    , choices=list_options
                    )
args = parser.parse_args()

# get list of input from parser
list_inputs = args.input
