"""
This model provides functions to interface with the external lambda calculus parser.
"""

import depccg.parsing

from depccg.instance_models import load_model
from depccg.lang import set_global_language_to
from depccg.annotator import english_annotator, annotate_XX, logger
from depccg.printer import to_string
from depccg.tree import *
from depccg.allennlp.utils import read_params


def setup_model(annotator: str = "spacy", language: str = "en", unary_penalty: float = 0.1,
                nbest: int = 1, pruning_size: int = 50, beta: float = 1e-4, use_beta: bool = True,
                max_length: int = 250, max_step: int = 1e7, num_processes: int = 4):
    set_global_language_to(language)
    annotator_function = get_annotator(annotator)
    supertagger, config = load_model("elmo")
    kwargs = dict(
        unary_penalty=unary_penalty,
        nbest=nbest,
        pruning_size=pruning_size,
        beta=beta,
        use_beta=use_beta,
        max_length=max_length,
        max_step=max_step,
        processes=num_processes,
    )

    return supertagger, config, annotator_function, annotator, kwargs


def get_annotator(annotator_variant: str):
    return english_annotator.get(annotator_variant, annotate_XX)


def get_lambda_of(
        input_str: str,
        supertagger,
        config,
        annotator_function,
        annotator,
        kwargs):
    apply_binary_rules, apply_unary_rules, category_dict, _ = read_params(config.config)
    root_categories = [Category.parse(category) for category in 'S[dcl]|S[wq]|S[q]|S[qem]|NP'.split('|')]
    semantic_templates = config.semantic_templates

    inp = [line for line in map(str.strip, input_str.split("\n"))]

    document = annotator_function([
        [word for word in sentence.split(" ")]
        for sentence in inp if len(sentence) > 0
    ], tokenize=annotator)

    logger.info("supertagging")
    score_result, categories_ = supertagger.predict_doc(
        [[token.word for token in sentence] for sentence in document]
    )

    categories = [Category.parse(category) for category in categories_]

    if category_dict is not None:
        document, score_result = depccg.parsing.apply_category_filters(
            document,
            score_result,
            categories,
            category_dict,
        )

    results = depccg.parsing.run(
        document,
        score_result,
        categories,
        root_categories,
        apply_binary_rules,
        apply_unary_rules,
        **kwargs,
    )

    return to_string(results, format="ccg2lambda", semantic_templates=semantic_templates)