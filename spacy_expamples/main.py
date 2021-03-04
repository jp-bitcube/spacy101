import random
import spacy
# from spacy.matcher import Matcher, PhraseMatcher
# from spacy.lang.en import English
from spacy.tokens import Doc, Span, Token
# from spacy.language import Language

nlp = spacy.load('en_core_web_md')

# # # Basic Lexical Properties of spacy model # # #
# doc = nlp('It costs $5.')
# print("Index: ", [token.i for token in doc])
# print("Text: ", [token.text for token in doc])
# print("is_alpha: ", [token.is_alpha for token in doc])
# print("is_punct: ", [token.is_punct for token in doc])
# print("like_num: ", [token.like_num for token in doc])

# # # Linguistic Features # # #
# doc = nlp('She ate the pizza!')
#
# for token in doc:
#     # he part-of-speech tag of the token head.
#     print('Part of speech', token.text, '-->', token.pos_)
#     # The syntactic relation connecting child to head.
#     print('Dependency parser', token.text, '-->', token.dep_)
#     # The original text of the token head.
#     print('Head Text', token.text, '-->', token.head.text)

# # # Predicting Named Entities # # #

# doc = nlp('Apple is looking at buying U.K. startup for $1 Billion')
#
# for ent in doc.ents:
#     print(ent.text, '-->', ent.label_)


# # # Explain Spacy Labels # # #

# print(spacy.explain('GPE'))
# print(spacy.explain('NNP'))
# print(spacy.explain('dobj'))

# # # Matcher Patterns # # #
# pattern = [
#     {'IS_DIGIT': True},
#     {'LOWER': 'fifa'},
#     {'LOWER': 'world'},
#     {'LOWER': 'cup'},
#     {'IS_PUNCT': True},
# ]
# pattern1 = [{'TEXT': 'iPhone'}, {'TEXT': 'X'}]
# pattern2 = [{'LOWER': 'iphone'}, {'LOWER': 'x'}]
# pattern3 = [
#     {'LEMMA': 'buy'},
#     {'POS': 'DET', 'OP': '?'},  # Optional: matches 0 - 1 times
#     {'POS': 'NOUN'}
# ]
# pattern4 = [
#     {'LEMMA': 'love', 'POS': 'VERB'},
#     {'POS': 'NOUN'}
# ]
#
# # # Using The Matcher # # #
#
# matcher = Matcher(nlp.vocab)
# matcher.add("PATTERN_NAME", [pattern, pattern1, pattern2, pattern3, pattern4])
# # doc = nlp('Upcoming iPhone X, has leaked the release date')
# # doc2 = nlp('2018 FIFA world cup: France won!')
# # doc3 = nlp('I loved dogs now I love cats more')
# doc4 = nlp('I have bought a smart phone. now I\'m buying apps')
#
# matches = matcher(doc4)
# for match_id, start, end in matches:
#     string_id = nlp.vocab.strings[match_id]  # Get string representation
#     span = doc4[start:end]  # The matched span
#     print(match_id, string_id, start, end, span.text)

# # #  StringStore  # # #

# coffee_hash = nlp.vocab.strings["coffee"]
# coffee_string = nlp.vocab.strings[coffee_hash]
#
# print(coffee_string, coffee_hash)

# # #  Lexemes  # # #

# doc = nlp('I love coffee')
# lexeme = nlp.vocab["coffee"]
#
# print(lexeme.text, lexeme.orth, lexeme.is_alpha)

# # #  Doc Object and Span object  # # #

# model = English()
#
# words = ['Hello', 'World', '!']
# spaces = [True, False, False]
#
# # Creating a doc manually
# doc = Doc(model.vocab, words=words, spaces=spaces)
#
# span = Span(doc, 0, 2)
#
# span_with_label = Span(doc, 0, 2, label="GREETING")
#
# doc.ents = [span_with_label]
#
# print(doc.ents)

# # # Similarity # # #

# # 2 documents
# doc1 = nlp('I like fast food')
# doc2 = nlp('I love pizza')
# print(doc1.similarity(doc2))
#
# # 2 tokens
# doc = nlp('I like pizza and pasta')
# print(doc[2].similarity(doc[4]))
#
# # document and token
# d = nlp('I love pizza')
# t = nlp('soap')[0]
#
# print(d.similarity(t))
#
# # span and document
# span = nlp('I like pizza and pasta')[2: 5]
# document = nlp('MacDonalds sells burgers')
#
# print(span.similarity(document))
# # Vectors
# print(document[2].vector)

# # # Adding Statistical predictions # # #

# matcher = Matcher(nlp.vocab)
# pattern = [
#     {'LOWER': 'golden'},
#     {'LOWER': 'retriever'},
# ]
# matcher.add('DOG', [pattern])
# doc = nlp("I have a Golden Retriever")
#
# for match_id, start, end in matcher(doc):
#     span = doc[start:end]
#     print('Matched Span: ', span.text)
#     # Get the span root token and root head
#     print('Root token: ', span.root.text)
#     print('Root head: ', span.root.head.text)
#     # Get the previous token and its POS tag
#     print("Previous token: ", doc[start-1].text, doc[start-1].pos_)

# # # Efficient Phrase Matcher # # #

# matcher = PhraseMatcher(nlp.vocab)
# pattern = nlp('Golden Retriever')
# matcher.add('DOG', [pattern])
# doc = nlp("I have a Golden Retriever")
#
# for match_id, start, end in matcher(doc):
#     span = doc[start:end]
#     print('Matched phrase: ', span.text)

# # # Pipeline attributes # # #

# print(nlp.pipe_names)
# print(nlp.pipeline)

# # # Simple Custom Pipeline Component # # #

# @Language.component("length")
# def custom_component(doc):
#     print('DOC LENGTH: ', len(doc))
#     return doc
#
#
# nlp.add_pipe('length', first=True)
# d = nlp("This is a sentence.")
#
# print('Pipelines', nlp.pipe_names, d)

# # # Setting Custom Attributes # # #

# Register on Doc, Span and Token objects
# Doc.set_extension("title", default=None)
# Span.set_extension("is_color", default=False)
# Token.set_extension("has_color", default=False)

# custom metadata to documents, span and tokens via the ._ attribute
# doc._.title = "My Document"
# doc._.is_color = True
# doc._.has_color = False

# example of a token extension
# def get_is_color(token):
#     colors = ['red', 'green', 'blue']
#     return token.text in colors
#
#
# Token.set_extension("is_color", getter=get_is_color)
#
# doc = nlp("The sky is blue")
#
# print(doc[3]._.is_color, "-->",  doc[3].text)

# example of a span extension
# def get_has_color(span):
#     colors = ['red', 'green', 'blue']
#     return any(token.text in colors for token in span)
#
#
# Span.set_extension("has_color", getter=get_has_color)
# doc = nlp("The sky is blue")
#
# print(doc[1:4]._.has_color, '-->', doc[1:4].text)
# print(doc[0:2]._.has_color, '-->', doc[0:2].text)

# def has_token(doc, token_text):
#     in_doc = token_text in [token.text for token in doc]
#     return in_doc
#
#
# Doc.set_extension('has_token', method=has_token)
# doc = nlp("The sky is blue")
#
# print(doc._.has_token("blue"), '--> blue')
# print(doc._.has_token("color"), '--> color')

# # # Scaling and Performance # # #

# Doc.set_extension("id", default=None)
# Doc.set_extension("page_number", default=None)
#
# data = [
#     ("This is a text", {"id": 1, "page_number": 15}),
#     ("And another text", {"id": 2, "page_number": 16})
# ]
#
# for doc, context in nlp.pipe(data, as_tuples=True):
#     doc._.id = context["id"]
#     doc._.page_number = context["page_number"]

# # Training a model
# model = spacy.blank("en")
# ner = model.create_pipe("ner")
# model.add_pipe(ner)
# ner.add_label("Gadget")
# examples = []
# nlp.begin_training()
#
# for itn in range(10):
#     random.shuffle(examples)
#
#     for batch in spacy.util.minibatch(examples, size=2):
#         texts = [text for text, annotation in batch]
#         annotations = [annotation for text, annotation in batch]
#
#         nlp.update(texts, annotations)
