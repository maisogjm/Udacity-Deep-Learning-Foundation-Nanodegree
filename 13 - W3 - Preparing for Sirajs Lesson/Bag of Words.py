from collections import Counter

def bag_of_words(text):
    # TODO: Implement bag of words
    total_counts = Counter()

    # Convert to lower case.
    for word in text.lower().split(" "):
        total_counts[word] += 1
    return(total_counts)

test_text = 'the quick brown fox jumps over the lazy dog'

print(bag_of_words(test_text))