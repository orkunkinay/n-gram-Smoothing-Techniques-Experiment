import nltk
import random
import numpy as np
from collections import defaultdict, Counter

nltk.download('brown')

class NGramModel:
    def __init__(self, n, smoothing='none', discount=0.75, lambdas=(0.7, 0.3)):
        self.n = n
        self.smoothing = smoothing
        self.discount = discount
        self.lambdas = lambdas
        self.ngrams = defaultdict(Counter)
        self.context_counts = defaultdict(int)
        self.lower_order_counts = Counter()
        self.vocab = set()
        self.UNK = '<UNK>'
    
    def preprocess(self, corpus):
        """Tokenize and handle unknown words"""
        tokenized_corpus = []
        word_counts = Counter(word.lower() for sentence in corpus for word in sentence)
        self.vocab = {word for word in word_counts if word_counts[word] > 1}
        self.vocab.add(self.UNK)
        
        for sentence in corpus:
            tokenized_sentence = []
            for word in sentence:
                word = word.lower()
                if word in self.vocab:
                    tokenized_sentence.append(word)
                else:
                    tokenized_sentence.append(self.UNK)
            tokenized_corpus.append(tokenized_sentence)
        
        return tokenized_corpus

    def train(self, corpus):
        tokenized_corpus = self.preprocess(corpus)
        for sentence in tokenized_corpus:
            padded_sentence = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            for i in range(len(padded_sentence) - self.n + 1):
                ngram = tuple(padded_sentence[i:i+self.n])
                context = ngram[:-1]
                word = ngram[-1]
                self.ngrams[context][word] += 1
                self.context_counts[context] += 1
                self.lower_order_counts[context[1:]] += 1
    
    def laplace_probability(self, context, word):
        context_tuple = tuple(context)
        return (self.ngrams[context_tuple][word] + 1) / (self.context_counts[context_tuple] + len(self.vocab))
    
    def good_turing_probability(self, context, word):
        context_tuple = tuple(context)
        count = self.ngrams[context_tuple][word]
        n = sum(self.context_counts.values())
        if count == 0:
            return self.ngrams[context_tuple][1] / n
        else:
            next_count = self.ngrams[context_tuple][count + 1]
            if next_count == 0:
                return count / n  # To avoid division by zero, fall back to count / n
            return ((count + 1) * next_count / self.ngrams[context_tuple][count]) / n
    
    def jelinek_mercer_probability(self, context, word):
        context_tuple = tuple(context)
        lambda1, lambda2 = self.lambdas
        count = self.ngrams[context_tuple][word]
        lower_order_ngram = context[1:]
        lower_order_prob = self.unsmoothed_probability(lower_order_ngram, word)
        total_count = self.context_counts[context_tuple]
        if total_count == 0:
            return lambda2 * lower_order_prob
        return lambda1 * (count / total_count) + lambda2 * lower_order_prob
    
    def kn_probability(self, context, word):
        context_tuple = tuple(context)
        if context_tuple in self.ngrams and word in self.ngrams[context_tuple]:
            count = self.ngrams[context_tuple][word]
        else:
            count = 0
        
        lower_order_prob = self.lower_order_counts[context_tuple[1:]] / len(self.ngrams)
        numerator = max(count - self.discount, 0) + self.discount * len(self.ngrams[context_tuple]) * lower_order_prob
        denominator = self.context_counts[context_tuple] + self.discount * len(self.ngrams[context_tuple])
        if denominator == 0:
            return 0  # Avoid division by zero
        return numerator / denominator
    
    def modified_kn_probability(self, context, word):
        context_tuple = tuple(context)
        count = self.ngrams[context_tuple][word]
        lower_order_ngram = context[1:]
        lower_order_prob = self.unsmoothed_probability(lower_order_ngram, word)
        total_count = self.context_counts[context_tuple]
        if total_count == 0:
            return lower_order_prob
        return max(count - self.discount, 0) / total_count + (self.discount / total_count) * lower_order_prob

    def unsmoothed_probability(self, context, word):
        context_tuple = tuple(context)
        if context_tuple in self.ngrams and word in self.ngrams[context_tuple]:
            if self.context_counts[context_tuple] == 0:
                return 0  # Avoid division by zero
            return self.ngrams[context_tuple][word] / self.context_counts[context_tuple]
        else:
            return 0.0
    
    def probability(self, context, word):
        if self.smoothing == 'laplace':
            return self.laplace_probability(context, word)
        elif self.smoothing == 'good_turing':
            return self.good_turing_probability(context, word)
        elif self.smoothing == 'jelinek_mercer':
            return self.jelinek_mercer_probability(context, word)
        elif self.smoothing == 'kneser_ney':
            return self.kn_probability(context, word)
        elif self.smoothing == 'modified_kneser_ney':
            return self.modified_kn_probability(context, word)
        elif self.smoothing == 'none':
            return self.unsmoothed_probability(context, word)
        else:
            raise ValueError(f"Unsupported smoothing technique: {self.smoothing}")
    
    def generate_sentence(self, length=10):
        context = ['<s>'] * (self.n - 1)
        sentence = []
        for _ in range(length):
            context_tuple = tuple(context)
            words = list(self.ngrams[context_tuple].keys())
            probabilities = [self.probability(context_tuple, word) for word in words]
            if sum(probabilities) == 0:
                break  # If all probabilities are zero, stop generating
            word = random.choices(words, weights=probabilities)[0]
            if word == '</s>':
                break
            sentence.append(word)
            context = context[1:] + [word]
        return ' '.join(sentence)
    
    def perplexity(self, corpus):
        """Compute perplexity for a given corpus"""
        tokenized_corpus = self.preprocess(corpus)
        log_prob = 0
        word_count = 0
        for sentence in tokenized_corpus:
            padded_sentence = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            for i in range(len(padded_sentence) - self.n + 1):
                ngram = tuple(padded_sentence[i:i+self.n])
                context = ngram[:-1]
                word = ngram[-1]
                probability = self.probability(context, word)
                if probability > 0:
                    log_prob += np.log(probability)
                else:
                    log_prob += np.log(1e-10)  # To avoid log(0) error
                word_count += 1
        return np.exp(-log_prob / word_count)
