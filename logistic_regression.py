import pandas as pd
from sklearn.linear_model import LogisticRegression

def process_data(filename):
    data=pd.read_csv(filename, sep=' ', header=None, names=['token', 'POS', 'chunking', 'chunking2'])
    data=data[['token','POS']]
    tags = data['POS'].unique()
    return data, tags

def main():
    data, tags = process_data('train.txt')
    add_features(data)
    print(data.head())

def add_features(data):
    '''
    Basic Feature Definitions:
        X- Capitalization              --> may be proper noun
        X- word length
        X- ends in "ly"                --> most likely an adverb
        X- "a", "an", "the"            --> DT
        X- ends in "ed"                --> most likely past tense verb
        X- ends in "tion" or "sion" or "ment"    --> most liekly a noun
        X- ends in "ies" or "xes"               --> most likely a plural noun
        X- ends in "er" or "or"        --> most likely a singular noun  e.g. "teacher"/"worker"/"lawyer"
        X- ends in "able" or "ible"    --> most likely an adjective     e.g. "likable"/""
        X- "and", "but", and "or"      --> CC
        X- ","                         --> ,
        X- "."                         --> .
        X- "$"                         --> $
        X- "''"                        --> ''
        X- contains number             --> CD
        X- "+" or "="                  --> SYM
        - after "is"                  --> likely a noun or adjective  
        - after "a"/"an"/"the"        --> likely a noun or adjective   e.g. "the dog"/"the happy dog"
        X- begins with "un"            --> likely an adjective          e.g. "unattractive"/"unlikely"
        X- begins with "in"/"il"/"im"  --> likely an adjective          e.g. "illegal"/"inactive"
        - begins with "dis" or "re"           --> likely a verb                e.g. "disconnect","dislike"
    '''
    data['capitalized'] = data['token'].apply(lambda x: 1 if x.istitle() else 0)
	# data['word_length'] = data['token'].apply(lambda x: len(x))
	# data['and_or_but'] = 1 if input_word in ['and', 'but', 'or'] else 0
	# data['a_an_the'] = 1 if input_word in ['a', 'an', 'the'] else 0
	# data['comma'] = 1 if input_word == ',' else 0
	# data['period'] = 1 if input_word == '.' else 0
	# data['dollar_sign'] = 1 if input_word == '$' else 0
	# data['single_quotes'] = 1 if input_word == "''" else 0
	# data['contains_number'] = 1 if any(chr.isdigit() for chr in input_word) else 0
	# data['plus_or_equals'] = 1 if input_word in ['+', '='] else 0
	# data['prefix'] = get_prefix_num(input_word)

if __name__ == '__main__':
    main()
