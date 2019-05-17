import random
import pandas as pd
from queue import CircularQueue
import nltk
#nltk.download()
from nltk.tree import ParentedTree as Tree
import en

class negation:

    def __init__(self,max_length):
        df = pd.read_csv(r'snli_1.0_train.txt', delimiter='\t')
        self.df = df[df['sentence1'].apply(lambda x: len(x) < max_length)]
        self.df.reset_index()
        self.entailment_df=df[df['gold_label']=='entailment']

    #breadth first search which returns the first
    #node that's in the list of search parameters
    def bfs(self,tree, search=None):
        l=len([sub for sub in tree.subtrees()])
        q = CircularQueue()

        if tree.label()=='ROOT':
            q.enqueue(tree[0])
        else:
            q.enqueue(tree)
        i=0
        while not q.is_empty():
            t = q.dequeue()
            if t.label() in search:
                return t
            for child in t:
                if isinstance(child, nltk.tree.Tree) and child.label()!='S':
                    q.enqueue(child)
            i+=1

            if i>l+1:
                print('BFS error')
                return False
        return False

    #Negates the main verb in a sentence
    def negate_verb(self,t):
        verbs = ['VBP', 'VBD', 'VBZ', 'VBN', 'VB', 'VBG']
        tr = self.bfs(t, ['VP'])
        if tr is False:
            print('tr false')
            return False
            # tr = self.bfs(t, ['NP'])
            # if tr is None:
            #     return None
            # else:
            #     tr.insert('RB', ['not'])

        for index2, c in enumerate(tr.subtrees()):
            if c.label() in verbs:
                print('input word:',c.leaves()[0])
                print('input label:',c.label())
                word_negation=self.word_negation(c.leaves()[0],c.label())
                print('word negation:',word_negation)
                if word_negation is None:
                    return False
                print('word negation:',word_negation)
                c[0]=word_negation
                break
        return True


    #Searches for an easily negated preposition
    def negate_noun(self,t):
        np=self.bfs(t,['NP'])
        if tr is None:
            return False
        if pp is None:
            return False
        pp=self.bfs(np,['PP'])
        for child in pp:
            if child.label()=='IN':
                word=child.leaves()[0]
                if word=='with':
                    child.leaves()[0]='without'
                else:
                    child.leaves()[0]='not '+child.label()
                return True
        return False

    #Uses NodeBox to return the present tense of a verb
    def present_tense(self,word):
        return en.verb.present(word)

    #Take a sentence and returns its negation
    #This is pretty much the main function
    def negate_sentence(self,t,prep=False):
        one=False
        b=False
        for index,c in enumerate(t.subtrees()):
            if c.label()=="S" and c!=t:
                b= (b or self.negate_verb(c))
                one=True
        if one is False:
            b=(b or self.negate_verb(t))
            #self.negate_noun(t)
        if prep:
            sub = n.bfs(t, ['NP'])
            if sub is not False:
                for i, child in enumerate(sub):
                    if child.label() == 'PP':
                        v=False
                        for grandchild in sub:
                            if grandchild.label()=='VP':
                                v=True
                        if not v:
                            del sub[i]
                        break
        return b

    #Takes a verb and returns its negation or None
    #if it cannot find it
    def word_negation(self, word, label):
        print('word:',word)
        print('label:',label)
        if label == 'VP':
            if word == 'was':
                return 'was not'
            elif word == 'is':
                return 'is not'
            elif word == 'did':
                return 'did not'
            elif word=='be':
                return "don't be"
            else:
                try:
                    tense=en.verb.tense(word)
                except:
                    tense=en.verb.tense(en.verb.infinitive(word))
                print('tense:', tense)
                if tense=='infinitive':
                    return 'do not '+ word
                elif tense=='past':
                    return 'did not '+present_tense(word)
                elif tense=='present participle':
                    return 'not '+word
                else:
                    return 'did not ' + present_tense(word)
        if label == 'VBZ':
            if word == 'is':
                return 'is not'
            else:
                return 'does not ' + self.present_tense(word)
        if label== 'VBP':
            if word=='have':
                print('happened')
                return 'have not'
            if word=='are':
                return 'are not'
            return 'do not ' + en.verb.infinitive(word)
        if label == 'VBG':
            if len(word) > 3 and word[-3:] == 'ing':
                return 'not ' + word
        if label == 'VBN':
            return 'not' + word
        if label=='VB':
            return 'do not '+ word

    def random_row(self):
        i = random.randint(1,self.entailment_df.shape[0])
        row = self.entailment_df.iloc[i]
        id=row['pairID']
        return id,row

    def negate_row(self,row,i):
        p_tree=Tree.fromstring(row['sentence1_parse'])
        h_tree=Tree.fromstring(row['sentence2_parse'])

        p_tree.pretty_print()
        h_tree.pretty_print()

        b=self.negate_sentence(p_tree)
        if b is False:
            return False
        b=self.negate_sentence(h_tree)
        if b is False:
            print("B FALSE")
            return False

        for sub in p_tree.subtrees():
            if sub.label()=='CC' or sub.label()=='PP':
                return False
        for sub in h_tree.subtrees():
            if sub.label()=='CC' or sub.label()=='PP':
                return False

        p_sent=row['sentence1']
        h_sent = row['sentence2']

        neg_p_sent=' '.join([word for word in p_tree.flatten()[:]])
        neg_h_sent = ' '.join([word for word in h_tree.flatten()[:]])

        return i,p_sent,h_sent,neg_p_sent,neg_h_sent

    def feeder(self):
        while True:
            i,row = self.random_row()
            print('row:',row['sentence1'])
            n=self.negate_row(row,i)
            print('n:',n)
            if not n is False:
                return n

    def contradiction_feeder(self):
        while True:
            i,row = self.random_row()
            print('row:',row['sentence1'])
            n=self.negate_row(row,i)
            print('n:',n)
            if not n is False:
                return n

    def create_df(self):
        df=pd.DataFrame({'sentence1':["first"],
                         'sentence2':["second"]})
        df.to_csv('contrapositives.csv')


    def add_contradiction_sentences(self):
        d = {}
        df = pd.read_csv(r'contradiction_train.csv')
        count = df.shape[0]
        for row in range(count):
            d[df.iloc[row]['index']] = 1
        a = True
        print("HELLO AND WELCOME")
        while a:
            try:
                i, p, h, neg_p, neg_h = self.feeder()
                if i in d:
                    print('REPEAT')
                    continue
            except KeyError:
                print('EXCEPTION OCCURRED')
                continue
            print('=' * 8)
            print('Original Premise:')
            print(p)
            print()
            print('Negated Premise:')
            print(neg_p)
            print()
            print()
            print('Original Hypothesis:')
            print(h)
            print()
            print('Negated Hypothesis:')
            print(neg_h)

            s = str(raw_input('Does this negation make sense?\n'))
            if s.strip() == 'yes':
                print('OG df:')
                print(df)
                print(df.columns)
                mini = pd.DataFrame({'sentence1': [p], 'sentence2': [h], 'index': [i],'sentence1_negation':[neg_p],'sentence2_negation':[neg_h]})
                print('mini:')
                print(mini)
                print(mini.columns)
                df = df.append(mini)
                ##Note that the append operation is not in-place
                print('df:')
                print(df)

                df.to_csv('contradiction_train.csv')
                count += 1
                print('count=', count)
            s = raw_input('Keep going?\n')
            if s.strip() == 'no':
                break


    def add_sentences(self):
        d={}
        df=pd.read_csv(r'contrapositives_test.csv')
        count=df.shape[0]
        for row in range(count):
            d[df.iloc[row]['index']]=1

        a=True
        print("HELLO AND WELCOME")
        while a:
            try:
                i, p, h, neg_p, neg_h=self.feeder()
                if i in d:
                    print('REPEAT')
                    continue
            except KeyError:
                print('EXCEPTION OCCURRED')
                continue
            print('='*8)
            print('Original Premise:')
            print(p)
            print()
            print('Negated Premise:')
            print(neg_p)
            print()
            print()
            print('Original Hypothesis:')
            print(h)
            print()
            print('Negated Hypothesis:')
            print(neg_h)

            s=str(raw_input('Does this negation make sense?\n'))
            if s.strip()=='yes':
                print('OG df:')
                print(df)
                mini=pd.DataFrame({'sentence1':[neg_h],'sentence2':[neg_p],'index':[i]})
                print('mini:')
                print(mini)
                df=df.append(mini)
                ##Note that the append operation is not in-place
                print('df:')
                print(df)

                df.to_csv('contrapositives_test.csv')
                count+=1
                print('count=',count)
            s=raw_input('Keep going?\n')
            if s.strip()=='no':
                break

    def negation(self,parse):
        neg_tree=Tree.fromstring(parse)
        self.negate_sentence(neg_tree)# YOUR NEGATION FUNCTION GOES HERE (parse)
        if neg_tree is False:
            return None
        return neg_tree.as_string()

    def save_ids(self, negated_data):
        full_data=self.entailment_df
        full_data['neg_2'] = full_data['sentence2_parse'].transform(negation)
        print('transformed')
        negated_data = negated_data.rename(columns={'sentence1': 'neg_2', 'sentence2': 'neg_1'})
        print('renamed')
        t = pd.merge(left=negated_data, right=full_data, how='inner')
        return t[['sentence1', 'sentence2', 'neg_1', 'neg_2', 'gold_label', 'pairID']]

if __name__=='__main__':

    n=negation(30)

    df=pd.read_csv(r'contrapositives.csv')
    print('started')
    h=n.save_ids(df)
    print(h)
    print(h.shape)
    #row=n.random_row()
    #n.add_contradiction_sentences()

#potential rules:
#If S in VP, then negate that VP too
#Add 'not' to IN within PP within NP