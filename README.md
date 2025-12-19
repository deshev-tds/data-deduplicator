# Hybrid deduplication engine
Why this is not “just" cosine similarity? 

This repository demonstrates a general approach to cleaning noisy, real-world datasets.
Although I used it to clean-up a noisy dataset for another project, it is NOT a recipe tool.

The actual problem I had was:
how to deduplicate semantically similar records without destroying meaningful variation.
(e.g. having 40,000 recipes in a dataset with the same names but different ingredients, the same ingredients but different names, 
very similar ingredients and names, but different prep methods, or different everything, but essentially the same dish. Mind -> blown.)

If your domain is job ads, research abstracts, product listings, support tickets, or scraped web content,
the same principles apply with minimal adaptation.

--------------------------------------------------
What problem this solves
--------------------------------------------------

In theory, semantic embeddings + cosine similarity should be enough.
In practice, they are not.

Real datasets suffer from:
- paraphrasing
- templated text
- boilerplate sections
- partial copies
- missing fields
- domain-specific noise

Cosine similarity alone will happily merge things that:
- describe different entities in similar language
- share structure but not meaning
- are semantically close but operationally distinct

This pipeline exists because naïve semantic deduplication breaks datasets silently.

--------------------------------------------------
Why cosine similarity alone is insufficient
--------------------------------------------------

Cosine similarity answers a fairly narrow question:
“How close are these two vectors in embedding space?”

It does NOT answer:
- Are these records functionally the same?
- Do they represent the same real-world entity?
- Is the overlap meaningful or accidental?

In noisy datasets, high cosine similarity often comes from:
- shared phrasing
- repeated instructions
- generic descriptions
- cultural or domain-specific boilerplates

If you deduplicate purely on cosine similarity, you will:
- lose legitimate variants
- collapse distinct items
- introduce subtle, hard-to-detect errors

These errors are worse than duplicates.

--------------------------------------------------
The hybrid approach
--------------------------------------------------

This "engine" uses a layered decision model:

1. Deterministic rules first
   Exact or near-exact matches are resolved cheaply and predictably.
   This removes obvious noise without touching semantics.

2. Semantic similarity second
   Sentence embeddings are used to propose candidate duplicates,
   not to make final decisions.

3. Domain signal as a filter
   A secondary signal (ingredients, attributes, features, tags)
   is required to confirm semantic similarity.

Only when BOTH:
- semantic similarity is high
- domain overlap is sufficient

do two records get treated as duplicates.

This mirrors how I, a human, reason about sameness - provided enough coffee has been part of the process. 

--------------------------------------------------
Why a graph, not pairwise deletion
--------------------------------------------------

Duplicates come in cluster, not pairs.

By modeling the dataset as a graph:
- nodes are records
- edges represent “likely duplicate” relationships

we can resolve entire connected components at once.

This avoids order-dependent deletion bugs and makes the logic auditable.

--------------------------------------------------
Survivor selection philosophy
--------------------------------------------------

When multiple records represent the same underlying entity,
the goal is not correctness in the abstract,
but information preservation.

The "survivor" is chosen to maximize:
- data completeness
- descriptive richness
- downstream utility

This is explicitly a data engineering decision, not a linguistic one.

--------------------------------------------------
Generalizing this beyond recipes
--------------------------------------------------

To adapt this pipeline to another domain, replace:

- description
  with the main free-text field you embed
- products_extracted
  with any structured or semi-structured signal that encodes substance
  (skills, features, tags, components, entities, attributes)
- ingredient overlap
  with an overlap or similarity function appropriate to your domain

Everything else remains the same.

--------------------------------------------------
What this is (not), conceptually
--------------------------------------------------

- not a classifier.
- not a magical deduplicator.
- not “Sam Altman cleaning your data with a magic wand or whatever”.

It is a way to combine:
- statistical similarity
- domain constraints
- deterministic rules

into a process that fails conservatively.

--------------------------------------------------
Why the example is recipes
--------------------------------------------------

Recipes are:
- multilingual
- heavily paraphrased
- semi-structured
- culturally noisy
- I needed those for a project anyway, so might as well. 

If this approach works for kyufteta, shopska salata & spicy rakia (Bulgarian things; don't ask)
it should work for most scraped human text.

--------------------------------------------------
How to use it
--------------------------------------------------

Read the script.
Change the domain signals.
Adjust the thresholds.
Inspect the output.

The code is intentionally explicit and linear.
Nothing is hidden, nothing is assumed (I think).
