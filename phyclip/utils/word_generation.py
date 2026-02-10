import random

# NLTK for word generation
try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    print("NLTK not found. Please install it: pip install nltk")
    exit()


def setup_nltk_data():
    """Downloads necessary NLTK data if not already present."""
    try:
        wn.ensure_loaded()
    except LookupError:
        print("Downloading NLTK data (wordnet, omw-1.4)...")
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        print("NLTK data downloaded.")
        # Re-initialize WordNet after download
        wn.ensure_loaded()


def get_root_synsets(level=1):
    """
    Returns a dictionary of named root synsets for WordNet.
    Can specify a level to get more specific synsets.
    """
    setup_nltk_data()

    base_synsets = {
        "physical_entity": wn.synset("physical_entity.n.01"),
        "abstract_entity": wn.synset("abstract_entity.n.01"),
        "living_thing": wn.synset("living_thing.n.01"),
        "artifact": wn.synset("artifact.n.01"),
        "event": wn.synset("event.n.01"),
        "food": wn.synset("food.n.01"),
        "location": wn.synset("location.n.01"),
    }

    if level <= 1:
        return base_synsets

    # Recursively find hyponyms to the specified level
    current_synsets = list(base_synsets.values())
    for _ in range(level - 1):
        next_level_synsets = []
        for s in current_synsets:
            if s is not None:
                next_level_synsets.extend(s.hyponyms())
        current_synsets = next_level_synsets

    # Create a dictionary with unique synset names
    # Filter out synsets with no further hyponyms to get meaningful groups
    # Also filter very generic names that might still exist at deeper levels
    generic_names = {"entity.n.01", "physical_entity.n.01", "abstract_entity.n.01"}

    final_synsets = {}
    for s in current_synsets:
        if s is not None and s.hyponyms() and s.name() not in generic_names:
            # Use a more descriptive name if possible
            name = s.name().split(".")[0]
            if name not in final_synsets:  # Avoid duplicate names
                final_synsets[name] = s

    # If the list is too large, select a subset to keep it manageable
    if len(final_synsets) > 20:
        # Sort by name and pick a subset for consistency
        sorted_keys = sorted(final_synsets.keys())
        final_synsets = {k: final_synsets[k] for k in sorted_keys[:20]}

    return final_synsets


def generate_wordnet_words(
    num_words=100, seed=42, root_synset=None, include_compounds=False, verbose=False
):
    """
    Generates a list of words using WordNet from noun synsets.
    If root_synset is provided, generates words from that specific synset's hyponyms.
    Otherwise, generates words with diverse concepts from a predefined list of roots.
    The seed is fixed to ensure the same list is generated every time.

    Args:
        num_words: Number of words to generate
        seed: Random seed for reproducibility
        root_synset: Specific synset to use as root (optional)
        include_compounds: Whether to include compound words (with spaces)
        verbose: Whether to print debug information
    """
    random.seed(seed)
    setup_nltk_data()

    if root_synset:
        # Use the provided synset as the only root
        root_synsets = [root_synset]
    else:
        # Get the standard list of diverse root synsets
        root_synsets = list(get_root_synsets().values())

    # Create a pool of all possible words from all noun synsets
    word_pool = []
    synset_count = 0

    for root in root_synsets:
        if root is not None:
            all_hyponyms = list(root.closure(lambda s: s.hyponyms()))
            # Include the root synset itself
            all_hyponyms.append(root)
        else:
            all_hyponyms = []

        if verbose:
            print(
                f"Root synset {root.name() if root else 'None'}: {len(all_hyponyms)} hyponyms"
            )

        for synset in all_hyponyms:
            # Include all noun synsets (.n.XX), not just .n.04
            synset_parts = synset.name().split(".")
            if len(synset_parts) >= 2 and synset_parts[1].startswith("n"):
                synset_count += 1
                for lemma in synset.lemmas():
                    word = lemma.name().replace("_", " ").lower()

                    # Filter based on compound word preference
                    if include_compounds or " " not in word:
                        word_pool.append(word)

    if verbose:
        print(f"Total synsets processed: {synset_count}")
        print(f"Total words before deduplication: {len(word_pool)}")

    # Ensure the list is unique
    word_pool = sorted(list(set(word_pool)))

    if verbose:
        print(f"Unique words: {len(word_pool)}")

    # If the pool is smaller than requested, return the whole pool
    if len(word_pool) <= num_words:
        if verbose:
            print(
                f"Returning all {len(word_pool)} words (less than requested {num_words})"
            )
        return word_pool

    # Randomly sample from the pool
    random.shuffle(word_pool)
    return word_pool[:num_words]
