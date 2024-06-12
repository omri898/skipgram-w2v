import pickle
import numpy as np
import re, random, math, collections, nltk
from nltk.corpus import stopwords
from nltk import skipgrams
nltk.download('punkt',quiet=True)
nltk.download('stopwords',quiet=True)


def normalize_text(fn):
    """ Loading a text file and normalizing it, returning a list of sentences.

    Args:
        fn: full path to the text file to process
    """
    # Read the file
    with open(fn, 'r') as file:
        text = file.read()

    # Lowercasing all text
    text = text.lower()

    # Replace newlines with whitespace
    text = text.replace('\n', ' ')  

    # Removing any special characters keeping only alphabets, numbers and dots
    pattern = "[\.\sA-Za-z0-9_-]*" 
    matches = re.findall(pattern, text) 
    text = ''.join(matches) 
    
    # Split the text into sentences
    sentences = nltk.sent_tokenize(text) 
    sentences = [sentence.rstrip('.') for sentence in sentences]

    # Remove stopwords from each sentence
    stop_words = set(stopwords.words('english'))
    filtered_sentences = []
    for sentence in sentences:
        words = sentence.split()
        filtered_words = [word for word in words if word not in stop_words]
        filtered_sentence = ' '.join(filtered_words)
        if filtered_sentence:  # Check if the filtered sentence is not empty
            filtered_sentences.append(filtered_sentence)

    return filtered_sentences

def sigmoid(x): return 1.0 / (1 + np.exp(-x))


def load_model(fn):
    """ Loads a model pickle and return it.

    Args:
        fn: the full path to the model to load.
    """
    with open(fn, 'rb') as file:
        sg_model = pickle.load(file)
    return sg_model

class SkipGram:
    def __init__(self, sentences, d=100, neg_samples=4, context=4, word_count_threshold=5):
        self.sentences = sentences
        self.d = d  # embedding dimension
        self.neg_samples = neg_samples  # num of negative samples for one positive sample
        self.context = context # the size of the context window (not counting the target word)
        self.word_count_threshold = word_count_threshold #ignore low frequency words (appearing under the threshold)

        self.word_count = collections.Counter() # a word:count dictionary
        self.word_index = {} # a word-index map
        self.index_word = {} # an index-word map
        self._build_vocabulary() # populates the word_count, word_index and index_word dictionaries

        self.vocab_size = len(self.word_count) # set to be the number of words in the model (how? how many, indeed?)

        # Initialize the embedding matrices
        self.T = None # target embedding matrix
        self.C = None  # context embedding matrix
        self.V = None # combined embedding matrix

    def _build_vocabulary(self):
        """Populates the word_count, word_index and index_word dictionaries."""
        # Count the frequency of each word in the sentences
        for sentence in self.sentences:
            words = sentence.split()
            self.word_count.update(words)

        # Filter out words that are under the threshold
        filtered_words = [word for word, count in self.word_count.items() if count >= self.word_count_threshold]
        self.word_count = {word: count for word, count in self.word_count.items() if count >= self.word_count_threshold}

        # Create a word-index map
        index = 0
        for word in filtered_words:
            self.word_index[word] = index
            index += 1

        # Create an index-word map
        self.index_word = {index: word for word, index in self.word_index.items()}

    def _cosine_similarity(self, v1, v2):
        """Returns the cosine similarity between two vectors.

        Args:
            v1: a numpy vector
            v2: a numpy vector
        """

        dot_product = np.dot(v1, v2)
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)

        if norm_v1 != 0 and norm_v2 != 0:
            return dot_product / (norm_v1 * norm_v2)
        return 0

    def compute_similarity(self, w1, w2):
        """ Returns the cosine similarity (in [0,1]) between the specified words.

        Args:
            w1: a word
            w2: a word
        Returns: a float in [0,1]; defaults to 0.0 if one of specified words is OOV.
    """
        sim = 0.0  # default
        
        if w1 not in self.word_index or w2 not in self.word_index:
            return sim

        w1_index = self.word_index[w1]
        w2_index = self.word_index[w2]

        # Retrieve the embeddings for the specified words
        w1_embedding = self.V[:, w1_index]
        w2_embedding = self.V[:, w2_index]

        # Calculate the cosine similarity
        sim = self._cosine_similarity(w1_embedding, w2_embedding)

        return sim

    def get_closest_words(self, w, n=5):
        """Returns a list containing the n words that are the closest to the specified word.

        Args:
            w: the word to find close words to.
            n: the number of words to return. Defaults to 5.
        """
        if w not in self.word_index:
            return []

        # Calculate similarity between w and all other words
        similarities = {}
        for other_word in self.word_index:
            if other_word != w:
                sim = self.compute_similarity(w, other_word)
                similarities[other_word] = sim

        # Sort words by similarity and return the top n words
        closest_words = sorted(similarities, key=similarities.get, reverse=True)[:n]

        return closest_words

    def _create_positive(self):
        """
        Creates positive samples for each target word in every sentence.
        
        Returns:
            A list of dictionaries. Each dictionary represents a sentence, with the target word as the key and a list of context words as the value.
        """

        def _build_contexts(samples):
            """
            Builds a context dictionary from positive skip-gram pairs.

            Args:
                samples: A list of tuples containing skip-gram pairs. Each tuple consists of (target_word, context_word).

            Returns:
                contexts: A dictionary where each key is a target word index and the value is a list of context word indexes.
                        {target_index_1: [context_index_1, context_index_2, ...],
                         target_index_2: [context_index_3, context_index_4, ...],
                         ...}
            """
            contexts = {}
            for target, context in samples:
                target_index = self.word_index[target]
                context_index = self.word_index[context]
                contexts.setdefault(target_index, []).append(context_index)
            return contexts
        
        total_positive_samples = []
        for sentence in self.sentences:
            # Filter out words that are not in the word_index
            sentence_list = [word for word in sentence.split(" ") if word in self.word_index]
            
            # Generate positive samples (forwards and backwards)
            positive = list(skipgrams(sentence_list, 2, self.context // 2 - 1)) + list(skipgrams(sentence_list[::-1], 2, self.context // 2 - 1))
            
            # Create a dictionary to store the target word and its context words
            contexts = _build_contexts(positive)
            
            # Add the dictionary to the positive_samples list if it contains any contexts
            if contexts:
                total_positive_samples.append(contexts)

        return total_positive_samples

    def _create_positive_negative(self, target_index, positive_indexes):
        """
        Adds random negative samples to each target word.

        Args:
            target_index (int): The index of the target word.
            positive_indexes (list): A list of indexes of positive context words.

        Returns:
            positive_negative_indexes (list): A list of positive and negative context indexes.
            y_true (numpy.ndarray): A vector of True labels for the positive and negative samples.
        """
        # Get the target word
        target_word = self.index_word[target_index]

        # Generate negative words
        # Select neg_samples negative samples for positive index randomly
        negative_words = random.choices(list(self.word_count.keys()), k=self.neg_samples * len(positive_indexes))
        negative_indexes = [self.word_index[word] for word in negative_words if word != target_word]
        # Combine positive and negative indexes
        positive_negative_indexes = positive_indexes + negative_indexes

        # Create the true labels
        y_true = np.concatenate((np.ones((len(positive_indexes), 1)), np.zeros((len(negative_indexes), 1))), axis=0)
        
        return positive_negative_indexes, y_true

    def _calculate_loss(self, y_pred, y_true):
        """
        Calculates binary cross entropy loss function.

        Parameters:
        - y_pred (numpy.ndarray): Predicted labels.
        - y_true (numpy.ndarray): True labels.

        Returns:
        - loss (float): The calculated loss value.
        """
        # Flatten the arrays
        y_pred_flat, y_true_flat = y_pred.flatten(), y_true.flatten()

        # Calculate the log probabilities for positive samples
        log_y_pred = np.log(y_pred_flat)
        positive_loss = np.dot(log_y_pred, y_true_flat.T)

        # Calculate the log probabilities for negative samples
        log_one_minus_y_pred = np.log(1 - y_pred_flat)
        negative_loss = np.dot(log_one_minus_y_pred, (1 - y_true_flat).T)

        # Combine the positive and negative losses and take the negative
        loss = -(positive_loss + negative_loss)

        return loss

    def learn_embeddings(self, step_size=0.001, epochs=50, early_stopping=3, model_path=None):
            """Returns a trained embedding models and saves it in the specified path

            Args:
                step_size: step size for  the gradient descent. Defaults to 0.0001
                epochs: number or training epochs. Defaults to 50
                early_stopping: stop training if the Loss was not improved for this number of epochs
                model_path: full path (including file name) to save the model pickle at.
            """
            # Initialize the embedding matrices
            vocab_size = self.vocab_size
            T = np.random.rand(self.d, vocab_size)  # embedding matrix of target words
            C = np.random.rand(vocab_size, self.d)  # embedding matrix of context words

            # Create positive training samples
            positive_samples = self._create_positive()

            # Initialize variables for tracking loss and early stopping
            losses = []
            not_improved_count = 0
            last_loss = math.inf

            # Training loop
            for epoch in range(epochs):
                epoch_loss = []
                for sentence_dict in positive_samples:
                    for target_index, positive_indexes in sentence_dict.items():
                        # Add negative samples to the training data
                        context_indexes, y_true = self._create_positive_negative(target_index, positive_indexes)
                        
                        # The number of positive and negative context words
                        total_pos_neg = len(context_indexes)

                        # Get the target word embedding
                        target_embedding = T[:, target_index][:, None] # column weight vector of the target word

                        # Get the context words embeddings
                        context_embeddings = C[context_indexes]

                        # Calculate the output layer and predicted values
                        output_layer = np.dot(context_embeddings, target_embedding) # similarity t*c
                        y_pred = sigmoid(output_layer)

                        # Calculate the error and loss
                        error = y_pred - y_true
                        loss = self._calculate_loss(y_pred, y_true)
                        # Update the epoch loss
                        epoch_loss.append(loss / total_pos_neg)

                        # Calculate the gradients
                        context_grad = np.dot(target_embedding, error.T).T
                        target_grad = np.dot(error.T, context_embeddings).T / total_pos_neg
                        # Update the embeddings
                        C[context_indexes, :] -= step_size * context_grad 
                        T[:, [target_index]] -= step_size * target_grad

                # Calculate the mean epoch loss and track the running loss
                avg_epoch_loss = np.average(epoch_loss)
                losses.append(avg_epoch_loss)

                # print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss}")

                # Check for early stopping if loss hasn't improved for the specified number of epochs in a row
                if last_loss < losses[-1]:
                    not_improved_count += 1
                    if not_improved_count >= early_stopping:
                        break
                else:
                    not_improved_count = 0
                last_loss = losses[-1]

                # Update the embeddings matrices after each epoch
                self.T, self.C = T, C
                self.V = self.combine_vectors(T, C, combo=2)

            # Save the model after finishing training
            if model_path:
                with open(model_path, "wb") as file:
                    pickle.dump(self, file)

    def combine_vectors(self, T, C, combo=0, model_path=None):
        """Returns a single embedding matrix and saves it to the specified path

        Args:
            T: The learned targets (T) embeddings (as returned from learn_embeddings())
            C: The learned contexts (C) embeddings (as returned from learn_embeddings())
            combo: indicates how wo combine the T and C embeddings (int)
                   0: use only the T embeddings (default)
                   1: use only the C embeddings
                   2: return a pointwise average of C and T
                   3: return the sum of C and T
                   4: concat C and T vectors (effectively doubling the dimention of the embedding space)
            model_path: full path (including file name) to save the model pickle at.
        """

        if combo == 0:
            V = T
        elif combo == 1:
            V = C
        elif combo == 2:
            V = (T + C.T) / 2
        elif combo == 3:
            V = T + C.T
        elif combo == 4:
            V = np.concatenate((T, C.T), axis=0)

        if model_path:
            with open(model_path, "wb") as file:
                pickle.dump(V, file)

        return V

    def find_analogy(self, w1,w2,w3):
        """Returns a word (string) that matches the analogy test given the three specified words.
           Required analogy: w1 to w2 is like ____ to w3.

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
        """
        # If the words are not in the vocabulary, return None
        if any(word not in self.word_index for word in [w1, w2, w3]):
            return None

        # Get the indexes and embeddings of the specified words 
        w1_index, w2_index, w3_index = self.word_index[w1], self.word_index[w2], self.word_index[w3]
        w1_embedding, w2_embedding, w3_embedding = self.V[:, w1_index], self.V[:, w2_index], self.V[:, w3_index]

        # Compute the target embedding
        target_embedding = w2_embedding - w1_embedding + w3_embedding

        # Find the word whose embedding is closest to the target embedding
        w = None
        best_similarity = -np.inf

        for word, _ in self.word_index.items():
            if word in [w1, w2, w3]:  # Skip the input words if they are in the vocabulary
                continue
            word_embedding = self.V[:, self.word_index[word]]
            similarity = self._cosine_similarity(word_embedding, target_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                w = word

        return w

    def test_analogy(self, w1, w2, w3, w4, n=1):
        """Returns True if sim(w1-w2+w3, w4)@n; Otherwise return False.
            That is, returning True if w4 is one of the n closest words to the vector w1-w2+w3.
            Interpretation: 'w1 to w2 is like w4 to w3'

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
             w4: forth word in the analogy (string)
             n: the distance (work rank) to be accepted as similarity
            """

        if any(word not in self.word_index for word in [w1, w2, w3, w4]):
            return False

        # Use find_analogy to get the vector for the analogy
        target_word = self.find_analogy(w1, w2, w3)

        if not target_word:
            return False

        # Use get_closest_words to find the n closest words to the target_word
        closest_words = self.get_closest_words(target_word, n)

        return w4 in closest_words